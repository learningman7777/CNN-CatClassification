"""
이미지를 전달하고, 직접 만든 모델에 맞게 데이터를 프리프로세싱 해주려면 custom python funcion을 만들어서 mlflow에 넣어주어야 합니다.
그 custom python function들을 제공하기 위한 파일입니다.
"""
import base64
import numpy as np
import pandas as pd
import os
import PIL
import yaml

import torch
from torchvision import transforms

import mlflow
import mlflow.pytorch
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.file_utils import TempDir

class CatPyfunc(object):
    """
    프리프로세싱 코드를 포함한 분류 모델
    
    base64로 인코딩된 image binary data를 입력 받아 프리프로세싱을 거치고 모델을 통과하여 결과를 dataframe으로 전달하는 역할을 합니다.
    """
    def __init__(self, model, domain):
        self._model = model
        self._domain = domain
        probs_names = ["p({})".format(x) for x in domain]
        self._column_names = ["predicted_label", "predicted_label_id"] + probs_names

    def predict(self, input):
        """
        예측 함수. mlflow에서 네트워크로 inference요청이 오면 자동으로 이 함수가 호출 됩니다.

        :param input:
        base64로 인코딩된 image binary data # base64.encodebytes(np.asarray(x)).tobytes())를 거친 이미지 텐서(x).
        
        :return: 
        pandas.DataFrame containing predictions with the following schema:
            Predicted class: string,
            Predicted class index: int,
            Probability(class==0): float,
            ...,
            Probability(class==N): float,

        [{"predicted_label": "Abyssinian", "predicted_label_id": "0", "p(Abyssinian)": "0.28483438", "p(basset_hound)": "-0.80232537"}]
        와 같은 형태로, 예측된 이미지 라벨, 라벨 아이디, 각 라벨일 확률을 담은 data frame.
        """
        
        probs = self._predict_images(input)
        m, _ = probs.shape
        label_idx = torch.argmax(probs, dim=1)
        label_idx = np.asarray(label_idx)
       
        labels = np.array([self._domain[i] for i in label_idx], dtype=np.str).reshape(m, 1)
        probs = probs.detach().numpy()
        output_data = np.concatenate((labels, label_idx.reshape(m, 1), probs), axis=1)
        print(output_data)
        res = pd.DataFrame(columns=self._column_names, data=output_data)
        res.index = input.index
        return res

    def _predict_images(self, images):
        """
        Generate predictions for input images.
        :param images: binary image data
        :return: predicted probabilities for each class
        """
        img_transform = transforms.Compose([
                        transforms.Resize((197, 197)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
       

        reconstructed_image_tensor_array = []
        for (image, shape) in zip(images['image'], images['shape']):
            imagebytes = bytearray(image, encoding="utf8")
            deserialized_bytes = base64.decodebytes(imagebytes)
            deserialized_bytes = np.frombuffer(deserialized_bytes, dtype=np.uint8)
            img_nparray = np.reshape(deserialized_bytes, newshape=shape)
            img = PIL.Image.fromarray(np.uint8(img_nparray), mode="RGB")
            imgTensor = img_transform(img)
            reconstructed_image_tensor_array.append(imgTensor)
        
        reconstructed_image_tensor = torch.stack(reconstructed_image_tensor_array, dim=0)
        with torch.no_grad():
            output = self._model(reconstructed_image_tensor).squeeze(-1).squeeze(-1)
        return output


def log_model(pytorch_model, artifact_path, domain):
    """
    CatPyfunc로 바로 사용할 수 있도록 모델과 추가적인 정보를 기록하는 함수입니다.

    :param pytorch_model: model to be saved.
    :param artifact_path: Run-relative artifact path this model is to be saved to.
    :param domain: 도메인. 모델이 예측해서 0,1로 결과를 냈을때, 이걸 스트링으로 매칭 시킬 수 있도록 하기위해 저장하는 정보.
    """

    with TempDir() as tmp:
        data_path = tmp.path("image_model")
        os.mkdir(data_path)
        conf = {
            "domain": "/".join(map(str, domain))
        }
        with open(os.path.join(data_path, "conf.yaml"), "w") as f:
            yaml.safe_dump(conf, stream=f)
        pytorch_path = os.path.join(data_path, "pytorch_model")
        mlflow.pytorch.save_model(pytorch_model, path=pytorch_path)
        conda_env = tmp.path("conda_env.yaml")
        with open(conda_env, "w") as f:
            f.write(conda_env_template.format(python_version=PYTHON_VERSION,
                                              pytorch_version=torch.__version__,
                                              pillow_version=PIL.__version__))

        mlflow.pyfunc.log_model(artifact_path=artifact_path,
                                loader_module=__name__,
                                code_path=[__file__],
                                data_path=data_path,
                                conda_env=conda_env)


def _load_pyfunc(path):
    """
    CatPyfunc 모델을 로드하는 함수입니다. mlflow로 serve하면 자동으로 호출됩니다.
    """
    with open(os.path.join(path, "conf.yaml"), "r") as f:
        conf = yaml.safe_load(f)
    pytorch_model_path = os.path.join(path, "pytorch_model")
    domain = conf["domain"].split("/")
    

    pytorch_model = mlflow.pytorch.load_model(pytorch_model_path, map_location={'cuda:0': 'cpu'})
    return CatPyfunc(pytorch_model, domain=domain)


conda_env_template = """
name: catclassification
channels:
  - defaults
  - anaconda
dependencies:
  - python=={python_version}
  - torch=={pytorch_version}
  
  - pip:
    - pillow=={pillow_version}
"""
