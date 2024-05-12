# Email Guardian ML

This simple project is Chrome Extensions focusing on utilizing machine learning for basic detection of phishing links, email spam content, and legitimate files , with the help of the frontend python server that runs the ML models.


## Installation

**Prerequisites:**

* Python 3.x (https://www.python.org/downloads/)
* requirements.txt

**Importing the Project:**

```bash
git clone 
pip install -r requirements.txt
```

**Installing the chrome extension :**
1. Open Extensions settings :
   a. if on Chrome : (`chrome://extensions/`)
   b. if on Edge : (`edge://extensions/`)
2. Enable Developer Mode.
3. Click "Load unpacked".
4. Select the extension folder (`project/browser_extensions`).
5. Confirm any warnings.
6. Extension is loaded and active.

**Running the python server :**
Simply run (`project/python_server/start_all.bat`) file to run all the 3 servers at once : 
- `phishing_detection/main.py` for phishing detection 
- `Email-Spam-Detection/main.py` for email spam detection 
- `Malware-detection/main.py` for malware detection 


## Usage


**Browser Extensions :**

- When opening an email, you can click the extensions popup.
- By clicking Verification button, the extensions will communicate with the backend, and gives you result.


