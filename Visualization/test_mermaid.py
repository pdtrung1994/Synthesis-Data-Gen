import base64
import urllib.request
graph='''flowchart TD
  A["$$ \\theta_{opt} $$"] --> B'''
b64=base64.urlsafe_b64encode(graph.encode('utf8')).decode('ascii')
req=urllib.request.Request('https://mermaid.ink/img/'+b64, headers={'User-Agent': 'Mozilla/5.0'})
open('test_math.png', 'wb').write(urllib.request.urlopen(req).read())
print('OK')
