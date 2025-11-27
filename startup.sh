#!/bin/bash
cd /home/site/wwwroot
python3 -m streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
