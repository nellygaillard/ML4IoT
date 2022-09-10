from DoSomething import DoSomething
import time
import json
from datetime import datetime

class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)

        timestamp = input_json['bt']
        measure = input_json['bn']
        now = datetime.fromtimestamp(float(timestamp))
        datetime_str = now.strftime("%d-%m-%Y %H:%M:%S")

        for event in input_json['e']:
            if event['n'] == 'Predicted':
                pred = event['v']
                unit = event['u']
            else:
                real = event['v']
        print(f"({datetime_str})\t{measure} Alert:\tPredicted={pred} {unit}\tActual={real} {unit}")


if __name__ == "__main__":
    test = Subscriber("subscriber alert")
    test.run()
    test.myMqttClient.mySubscribe("/282613/TH_alert")

    while True:
        time.sleep(1)

