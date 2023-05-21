# for use if wanting a local whl in the python layer
FROM public.ecr.aws/sam/build-python3.10

# Copy modules whl
RUN echo "Copying data module whl..."
COPY ./*.whl /tmp/
RUN ls .
RUN pwd
