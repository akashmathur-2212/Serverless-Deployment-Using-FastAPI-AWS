FROM public.ecr.aws/amazonlinux/amazonlinux:2023

WORKDIR /app

# Install software
RUN yum update -y && \
    yum install -y python39 && \
    python3 -m ensurepip --upgrade && \
    python3 -m pip install --upgrade setuptools

COPY ./requirements.txt /app/requirements.txt
COPY ./artifacts /app/artifacts
COPY ./app.py /app/app.py
COPY ./utils.py /app/utils.py

RUN python3 -m pip install -r requirements.txt

RUN echo "Image Built"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile) #"python3", "-u"
CMD ["app.handler"]
