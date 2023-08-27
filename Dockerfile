FROM public.ecr.aws/lambda/python:3.10

WORKDIR /var/task

ENV HNSWLIB_NO_NATIVE=1

COPY . .

# RUN yum install python3-dev
RUN yum install -y python3-devel.x86_64
# RUN yum install build-essential -y
RUN yum install gcc gcc-c++ kernel-devel make -y

RUN pip install -r requirements-torch-inference.txt

# Copy modules whl
# RUN echo "Copying data module whl..."
# COPY ./*.whl /tmp/
# RUN ls .
# RUN pwd

CMD ["src.handlers.sec.chat"]
