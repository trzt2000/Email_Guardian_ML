@echo off
start wt -w 0 nt --title "Phishing detection" -d "./phishing_detection" cmd /k "python main.py"
ping localhost -n 2 >nul
start wt -w 0 nt --title "Email-Spam-Detection" -d "./Email-Spam-Detection" cmd /k "python main.py"
ping localhost -n 2 >nul
start wt -w 0 nt --title "Malware-detection" -d "./Malware-detection" cmd /k "python main.py"