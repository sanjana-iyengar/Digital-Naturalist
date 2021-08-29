import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("model.h5")

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        preds = model.predict(x)
        
        print("prediction",preds)
            
        index = ["The predicted animal is gatto , which belongs to cat family "
               "The predicted animal is pecora" ,
               "The predicted animal is mucca" ,
               "The predicted bird is antbird" ,
               "The predicted bird is peacock" ,
               "The predicted bird is wild turkey" ,
               "The predicted flower is rose" ,
               "The predicted flower is sunflower" ,
               "The predicted flower is tulip"]
 
        
        print(np.argmax(preds))
        
        text = "the predicted is : " + str(index[np.argmax(preds)])
        
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    