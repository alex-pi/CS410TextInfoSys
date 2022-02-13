import subprocess

process = subprocess.Popen('scrapy crawl links -a start_url=https://www.illinois.edu',
                           shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE )

print('===========')
#result = []
#for line in process.stdout:
#    result.append(line)
#errcode = process.returncode
#for line in result:
#    print(line)