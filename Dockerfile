FROM python:3.11

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements_cpu.txt

ENV data_url=''
ENV dry_run=false
ENV model_url='gs://news-crawled/model.pth'

CMD ["sh", "-c", "python predict.py --data_url=${data_url:-gs://news-crawled/$(TZ='America/Los_Angeles' date +'%m-%d')/data.csv} --model_url=${model_url} --dry_run=${dry_run}"]