# Tech Challenge Project

Tech Challenge is the project that encompasses the knowledge gained in all disciplines of the phase.
This project was bootstrapped with [Python](https://www.python.org/).

## Available Scripts

In the project directory, you can run:

### `python3 -m venv env`

The Python Virtual Environment (VENV) is used to execute and test scripts with various versions or dependencies in a specific environment. The scripts contained in the environment are all Python scripts which can be viewed under Scripts in the left navbar of Automation Gateway

### `source env/bin/activate`

Using the command source venv/bin/activate while in that directory, will activate that virtual environment

### `pip install <package_name>`

The pip install command is used to install software packages for Python from the Python Package Index (PyPI) or other package repositories. pip is the standard package manager for Python and is included with most modern Python distributions.

### `pip freeze > requirements.txt`

this command is used to create a file named requirements.txt that lists all the Python packages installed in the current environment, along with their specific versions.

### `python app.py`

The command python my_script.py is used to execute a Python program from the command line. The .py file extension is the standard convention for saving files that contain Python source code.

## Run the project on your machine

### `Run commands`
First you need to treaning and create the template files:
```shell script
python train_model.py
```
After create scaler.pkl and diabetes_model.pkl files execute the command below:
```shell script
python app.py
```
### `Test with Swagger`
Open your browser and past the URL: http://127.0.0.1:8090

### `Test with Curl`
On terminal, execute the code below
```shell script
curl -X 'POST' \
  'http://127.0.0.1:8090/Diabetes/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "pregnancies": 0,
  "glucose": 0,
  "bloodPressure": 0,
  "skinThickness": 0,
  "insulin": 0,
  "bmi": 0,
  "diabetesPedigreeFunction": 0,
  "age": 0
}'
```
## Run the project on Docker
### `Run commands`
First you need to treaning and create the template files:
```shell script
python train_model.py
```
### `Create a docker`
Create a docker:
```shell script
docker build -t diabetes:last .
```
### `Run docker`
Run the created docker:
```shell script
docker run --publish 8090:8090 diabetes:last
```
### `Test with Swagger`
Open your browser and past the URL: http://127.0.0.1:8090

### `Test with Curl`
On terminal, execute the code below
```shell script
curl -X 'POST' \
  'http://127.0.0.1:8090/Diabetes/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "pregnancies": 0,
  "glucose": 0,
  "bloodPressure": 0,
  "skinThickness": 0,
  "insulin": 0,
  "bmi": 0,
  "diabetesPedigreeFunction": 0,
  "age": 0
}'
```