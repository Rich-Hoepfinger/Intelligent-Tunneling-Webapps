#!/usr/bin/env python
# coding: utf-8

# In[4]:
from flask import Flask, request, render_template, Response
from tensorflow import keras
from keras.models import load_model
import tensorflow_addons as tfa
import numpy as np
import sklearn as sk
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
app = Flask(__name__)
d_b,density,E,v,phi,psi,c,img_str = 0,0,0,0,0,0,1,None
def ML_velo(file):
    global d_b,density,E,v,phi,psi,c,img_str
    ANN1 = 'Final_ANN1.h5'
    ANN2 = 'Final_NN2.h5'
    model1 = load_model(ANN1,custom_objects={'RSquare' : tfa.metrics.RSquare})
    model2 = load_model(ANN2,custom_objects={'RSquare' : tfa.metrics.RSquare})
    scaler_x = pd.read_pickle('scale_ann2.pkl')
    scaler_y = pd.read_pickle('scale_ann2_y.pkl') 
    scaler = pd.read_pickle('scale_ann1.pkl')
    def ANN2_predict(model2,scaler_x,scaler_y,file):
        a2 = scaler_y.inverse_transform(model2.predict(scaler_x.transform(file)))
        a2 = np.reshape(a2,(7,1))
        d_b,density,E,v,phi,psi,c = a2
        return d_b,density,E,v,phi,psi,c
    d_b,density,E,v,phi,psi,c = ANN2_predict(model2,scaler_x,scaler_y,file)

    def ANN1_Predict(model1,scaler,d_b,density,E,v,phi,psi,c):
        a1 = model1.predict(scaler.transform([[d_b[0],density[0],E[0],v[0],phi[0],psi[0],c[0]]]))
        a1 = np.reshape(a1,-1)
        return a1
#file=np.load(data_nn2.npy)
#make another method for the other output_ANN1(file, df_y1_5=pd.read_csv('cord.csv')), everything else is same
    def npy_output_ANN1(npy_file, df_y1_5=pd.read_csv('cord.csv')):
        a1 = npy_file
        df_out  = df_y1_5.dropna()
        a1 = np.reshape(a1,-1)
        df_out.insert(3,"ML",a1)
        df_y1_5 = df_y1_5.drop('S-S22', axis =1)
        df_out = df_out.drop('S-S22', axis =1)
        df_out_final = df_y1_5.merge(df_out, how = 'outer')
        return df_out_final

    ANN1_prediction = ANN1_Predict(model1,scaler,d_b,density,E,v,phi,psi,c)
    def output_ANN1(ANN1_prediction,df_y1_5=pd.read_csv('cord.csv')):
        a1 = ANN1_prediction
        df_out  = df_y1_5.dropna()
        df_out.insert(3,"ML",a1)
        df_y1_5 = df_y1_5.drop('S-S22', axis =1)
        df_out = df_out.drop('S-S22', axis =1)
        df_out_final = df_y1_5.merge(df_out, how = 'outer')
        #df_out_final = df_out_final.fillna(0)
        return df_out_final

    def predictions():
        global img_str
        ANN1_Post = output_ANN1(ANN1_Predict(model1,scaler,d_b,density,E,v,phi,psi,c))
        npy_Post = npy_output_ANN1(file)
        
        x = ANN1_Post['x'].values
        y = ANN1_Post['y'].values
        z = ANN1_Post['ML'].values
        xi = np.linspace(np.min(x), np.max(x), len(x))
        yi = np.linspace(np.min(y), np.max(y), len(y))
        X,Y= np.meshgrid(xi,yi)
        Z = griddata((x, y), z, (X, Y),method='nearest')
        fig, ax = plt.subplots()
        
        x_npy = npy_Post['x'].values
        y_npy = npy_Post['y'].values
        z_npy = npy_Post['ML'].values
        xi_npy = np.linspace(np.min(x_npy), np.max(x_npy), len(x_npy))
        yi_npy = np.linspace(np.min(y_npy), np.max(y_npy), len(y_npy))
        X_npy,Y_npy= np.meshgrid(xi_npy,yi_npy)
        Z_npy = griddata((x_npy, y_npy), z_npy, (X_npy, Y_npy),method='nearest')
        npy_fig, npy_ax = plt.subplots()
        
        vmin = min(min(z),min(z_npy))
        levels = np.arange(vmin, 0 + 0.04, 0.04)
        contour = ax.contourf(X,Y,Z, levels=levels)
        npy_contour = npy_ax.contourf(X_npy,Y_npy,Z_npy, levels=levels)
        # vmin, vmax = np.min(contour.cvalues), np.max(contour.cvalues)
        # npy_vmin, npy_vmax = np.min(npy_contour.cvalues), np.max(npy_contour.cvalues) 
        # contour.collections[0].set_clim(vmin=min(vmin, npy_vmin), vmax=max(vmax, npy_vmax))
        # npy_contour.collections[0].set_clim(vmin=min(vmin, npy_vmin), vmax=max(vmax, npy_vmax))
        
        npy_ax.set_title("$\sigma_y$ RNN")
        npy_fig.colorbar(npy_contour)
        npy_canvas=npy_fig.canvas
        npy_img_buffer = io.BytesIO()
        npy_canvas.print_png(npy_img_buffer)
        npy_img_str = base64.b64encode(npy_img_buffer.getvalue()).decode('utf-8')
        
        ax.set_title("$\sigma_y$ ANN")
        fig.colorbar(contour)
        canvas=fig.canvas
        img_buffer = io.BytesIO()
        canvas.print_png(img_buffer)
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return img_str,npy_img_str
    return predictions()   

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("webapp.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        file = request.files['file']
        file = np.load(file)
        img_str,npy_img_str = ML_velo(file)
        # put coheasion in kPa and density in tons/m3
    return render_template("webappSubmitted.html", npy_img_str=npy_img_str, img_str=img_str,d_b=str(round(d_b[0],2)),
                           density=str("{:0.2e}".format(density[0])), E=str(round(E[0],2)),v=str(round(v[0],2)),
                           phi=str(round(phi[0],2)), psi=str(round(psi[0],2)), c=str(round(c[0],2)))  
    
if __name__ =='__main__':
    app.run(debug = False)

# %%
