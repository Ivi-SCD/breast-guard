FROM python:3.10
WORKDIR /breast-guard
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt 
RUN pip install scikit-learn
COPY static ./static
COPY templates ./templates
COPY model ./model
COPY app.py .
EXPOSE 5000
CMD ["python", "app.py"]