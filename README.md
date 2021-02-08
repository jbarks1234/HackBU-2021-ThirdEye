# HackBU-2021-ThirdEye

### Steps to start the application

- Clone the repository (`git clone https://github.com/jbarks1234/HackBU-2021-ThirdEye.git`)
- `cd HackBU-2021-ThirdEye`

Using python environment -

- `python -m venv venv`
- `venv\Scripts\activate` if Windows or `source venv/bin/activate` if other
- `pip install flask torch image torchvision`
- `flask run`

Using Docker -

- Build the image using `docker build -t thirdeye .`
- Run the container using `docker run -d -p 80:5000 --name=thirdeye thirdeye` . This will start the app on a *development* server on port 80 of teh host. 