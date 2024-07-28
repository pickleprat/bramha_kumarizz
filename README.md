# Steph to run the module 

* Create a directory called "murli" and unzip the zip file you sent me into this directory 

     ```unzip [PATH TO ZIP FILE] -d ./murli``` 
* Create a .env file and define the variable ```MISTRAL_API_KEY``` inside it.

* Time to install the dependencies via ```pip install -r requirements.txt```

* Then run ```python app.py``` and the application will start generating vector db and after a long or short wait, your application server will start at localhost:4000. 

You can make requests to the api endpoint ```http://localhost:4000/generate/``` or simply run ```python test.py``` on another terminal window while keeping the server running. 

