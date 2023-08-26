FROM public.ecr.aws/lambda/python:3.10

WORKDIR /var/task

COPY . .
RUN pip install -r requirements-torch-inference.txt

# Copy modules whl
# RUN echo "Copying data module whl..."
# COPY ./*.whl /tmp/
# RUN ls .
# RUN pwd

CMD ["src.handlers.sec.chat"]
