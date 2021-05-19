from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd 

# d1=pd.read_csv("adjjj-c8739-default-rtdb-export (2).csv")
# d1=np.array(d1)
# X=d1[0][0]
# y=d1[0][1]

gsheetid = "1lA027q2n8QCEH5CERkcztzyG7mSWHqMkTx4CBLK5TGY"
sheet_name = "Sheet1"
gsheet_url = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(gsheetid, sheet_name)
df = pd.read_csv(gsheet_url,index_col=False)

gsheetid1 = "18spSY7iiWZuZcFH8G9GqvQ5D--rzKaMASAGUGJh10bY"
sheet_name1 = "Sheet1"
gsheet_url1 = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(gsheetid1, sheet_name1)
df1 = pd.read_csv(gsheet_url1, index_col=False)

d1=pd.merge(df1, df, left_index=True, right_index=True)
a=d1.tail(1)
a=np.array(a)
X=a[0][0]
y=a[0][1]



model=pickle.load(open('logisticrefrain.pkl','rb'))

app=Flask(__name__,template_folder='Template')


@app.route('/')
def man():
    return render_template("rain.html")


@app.route('/predict',methods=['POST'])
def home():
    a=request.form['Temperature']
    b=request.form['Humidity']
    arr=np.array([[a,b]])
    pred=model.predict(arr)

    # output='{0:.{1}f}'.format(pred[0][1], 2)
    output=pred
    return render_template('rainafter.html',data=output)

@app.route('/predict1',methods=['POST'])
def fb():
    arr1=np.array([[X,y]])
    # arr1=arr1.reshape(-1,1)
    pred=model.predict(arr1)

    # output='{0:.{1}f}'.format(pred[0][1], 2)
    output=pred
    return render_template('rainafter.html',data=output)


if __name__=="__main__":
    app.run(debug=True)
