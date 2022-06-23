FROM public.ecr.aws/lambda/python:3.8

COPY src/numpy-benchmark.py ${LAMBDA_TASK_ROOT}
COPY src/requirements.txt  .

RUN  pip3 install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD [ "numpy-benchmark.handler" ]
