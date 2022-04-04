from asyncio import sleep
from flask import Flask
from flask import request
from flask import render_template
import subprocess
import time
import json
import io
import pickle
from keras.models import load_model
import sys, errno
import asyncio
from flask_socketio import SocketIO, emit
from numpy import double
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import csv

app = Flask(__name__)
socketio = SocketIO(app)

# filename = 'decision_tree_mlp.sav'
filename='best_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
# loaded_model = load_model('hyprid_model.h5')
# loaded_model = load_model('nn_model_1.h5')

def abar(data):
    fname = 'test2.py'
    with open(fname, 'w') as f:
        f.write(data)
    obj=None
    result=None
    try:
        result = subprocess.run(['multimetric', 'test2.py'], stdout=subprocess.PIPE)
    except:
            print("EEEEEEEEEEEEEEEEEEEEEEE=>>>>")
            obj="Defect"
            result=None  
            ### Handle the error ###
    if obj!="Defect" and result!=None:
        try:
            fix_bytes_value = result.stdout.replace(b"'", b'"')
            obj = json.load(io.BytesIO(fix_bytes_value))
            obj=obj['overall']
        except:
            print("EEEEEEEEEEEEEEEEEEEEEEE2222=>>>>")
            obj="Defect"  
    # try:
    #     result = subprocess.run(['multimetric', 'test2.py'], stdout=subprocess.PIPE)
    #     fix_bytes_value = result.stdout.replace(b"'", b'"')
    #     obj = json.load(io.BytesIO(fix_bytes_value))
    #     obj=obj['overall']
    # except:
    #     obj="Defect"   
    return obj


@app.route('/')
def index():
    return render_template('index.html', result = {"code_metrics":"","defects":""})




@socketio.on('predict')
def handle_message(data):
            sample=data['data'].split(",")
            # sample = [float(numeric_string) for numeric_string in sample]

            # scaler = StandardScaler()
            # scaler.fit([sample])
            # sample = scaler.transform([sample])
            # sample =  pd.DataFrame(data=normalize(sample))
            result=loaded_model.predict([sample])
            item=None
            if result == True:
                item="Defected"
            else:
                item="Not Defected"
            emit("response",{"defects":str(item)})

            # item=None
            # with open('datasets/cm1.csv','rt')as f:
            #     data = csv.reader(f)
            #     for row in data:
            #         found=True
            #         item=row[21]
            #         # print(item)
            #         for i in range(21):
            #             if str(row[i])!=str(sample[i]):
            #                 found=False
            #                 item=None
            #         if found==True:
            #             print(found)
            #             emit("response",{"defects":str(item)})
            #             break


                        


            # result=np.argmax(result,axis=1)
            # result = (loaded_model.predict([sample]) > 0.5).astype("int32")








@socketio.on('extract')
def handle_message_extract(data):
    obj=(abar(data['data']))
    if obj!="Defect":
            emit("codeResponse",{"code_metrics":str(obj)})
    print('received message:'+str(data))




# [9, 1,1,1 37, 148.0, 37, 10.285714285714285, 14.388888888888891, 1522.2857142857142, 0, 84.57142857142857, 9, 0.0, 0, 9, 9, 7, 21, 16]

# sample= [9, 1,1,1, 37, 148.0, 37, 10.285714285714285, 14.388888888888891, 1522.2857142857142, 0, 84.57142857142857, 9, 0.0, 0, 9, 9, 7, 21, 16,1]
# sample=['1.1', '1.4', '1.4', '1.4', '1.3', '1.3', '1.3', '1.3', '1.3', '1.3', '1.3', '1.3', '2', '2', '2', '2', '1.2', '1.2', '1.2', '1.2', '1.4']
# sample = [float(numeric_string) for numeric_string in sample]
# result=loaded_model.predict([sample])
# print(result)




# %      1. loc             : numeric % McCabe's line count of code
# %      2. v(g)            : numeric % McCabe "cyclomatic complexity"
# %      3. ev(g)           : numeric % McCabe "essential complexity"
# %      4. iv(g)           : numeric % McCabe "design complexity"
# %      5. n               : numeric % Halstead total operators + operands
# %      6. v               : numeric % Halstead "volume"
# %      7. l               : numeric % Halstead "program length"
# %      8. d               : numeric % Halstead "difficulty"
# %      9. i               : numeric % Halstead "intelligence"
# %     10. e               : numeric % Halstead "effort"
# %     11. b               : numeric % Halstead 
# %     12. t               : numeric % Halstead's time estimator
# %     13. lOCode          : numeric % Halstead's line count
# %     14. lOComment       : numeric % Halstead's count of lines of comments
# %     15. lOBlank         : numeric % Halstead's count of blank lines
# %     16. lOCodeAndComment: numeric
# %     17. uniq_Op         : numeric % unique operators
# %     18. uniq_Opnd       : numeric % unique operands
# %     19. total_Op        : numeric % total operators
# %     20. total_Opnd      : numeric % total operands
# %     21: branchCount     : numeric % of the flow graph



if __name__ == '__main__':  
#    app.run(debug = True,threaded=True)  
    socketio.run(app)


# from gevent.wsgi import WSGIServer
# http_server = WSGIServer(('', 5000), app)
# http_server.serve_forever()
