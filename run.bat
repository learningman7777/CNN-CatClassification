set root=C:\Users\<user-name>\Anaconda3\envs\venv37
call C:\Users\<user-name>\Anaconda3\Scripts\activate.bat %root%
mlflow run ./ --no-conda
pause