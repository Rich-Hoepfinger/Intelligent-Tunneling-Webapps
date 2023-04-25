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
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

depth, list_Elas, list_density, list_v, list_phi, list_psi, list_c, normal_graph, predicted_graph, properties_graph, step_length, box_graph = [], [], [], [] ,[] ,[], [], None, None, None, None, None

app = Flask(__name__)
def ANN1_Predict(model1,scaler,d_b,density,E,v,phi,psi,c):
    a1 = model1.predict(scaler.transform([[d_b,density,E,v,phi,psi,c]]))
    a1 = np.reshape(a1,-1)
    return a1

def data_ses (step_length, elas_params, density_params, v_params, phi_params, psi_params, c_params):
    global depth, list_Elas, list_density, list_v, list_phi, list_psi, list_c, normal_graph
    half = step_length
    depth = []
    depth[0:half] = np.linspace(3,10,half)
    depth[half:2*half] = np.linspace(10,3,half)
    no = len(depth)
    list_Elas = []
    list_density = []
    list_v =[]
    list_phi = []
    list_psi =[]
    list_c =[]
    
    elas_mus, elas_sigmas = elas_params

    density_mus, density_sigmas = density_params 

    v_mus, v_sigmas =  v_params

    phi_mus, phi_sigmas = phi_params

    psi_mus, psi_sigmas = psi_params

    c_mus, c_sigmas = c_params

    for d_b in depth:
        if d_b >=3 and d_b<4:
            k = 0 
            mu_density,mu_elas,mu_v,mu_phi,mu_psi,mu_c = density_mus[k],elas_mus[k],v_mus[k],phi_mus[k],psi_mus[k],c_mus[k]
            sigma_density, sigma_elas, sigma_v, sigma_phi, sigma_psi, sigma_c = density_sigmas[k],elas_sigmas[k],v_sigmas[k],phi_sigmas[k],psi_sigmas[k],c_sigmas[k]
            
        if d_b >=4 and d_b<5:
            k = 1
            mu_density,mu_elas,mu_v,mu_phi,mu_psi,mu_c = density_mus[k],elas_mus[k],v_mus[k],phi_mus[k],psi_mus[k],c_mus[k]
            sigma_density, sigma_elas, sigma_v, sigma_phi, sigma_psi, sigma_c = density_sigmas[k],elas_sigmas[k],v_sigmas[k],phi_sigmas[k],psi_sigmas[k],c_sigmas[k]
        if d_b >=5 and d_b<6:
            k = 2 
            mu_density,mu_elas,mu_v,mu_phi,mu_psi,mu_c = density_mus[k],elas_mus[k],v_mus[k],phi_mus[k],psi_mus[k],c_mus[k]
            sigma_density, sigma_elas, sigma_v, sigma_phi, sigma_psi, sigma_c = density_sigmas[k],elas_sigmas[k],v_sigmas[k],phi_sigmas[k],psi_sigmas[k],c_sigmas[k]
        if d_b >=6 and d_b<7:
            k = 3 
            mu_density,mu_elas,mu_v,mu_phi,mu_psi,mu_c = density_mus[k],elas_mus[k],v_mus[k],phi_mus[k],psi_mus[k],c_mus[k]
            sigma_density, sigma_elas, sigma_v, sigma_phi, sigma_psi, sigma_c = density_sigmas[k],elas_sigmas[k],v_sigmas[k],phi_sigmas[k],psi_sigmas[k],c_sigmas[k]
            
        if d_b >=7 and d_b<8:
            k = 4 
            mu_density,mu_elas,mu_v,mu_phi,mu_psi,mu_c = density_mus[k],elas_mus[k],v_mus[k],phi_mus[k],psi_mus[k],c_mus[k]
            sigma_density, sigma_elas, sigma_v, sigma_phi, sigma_psi, sigma_c = density_sigmas[k],elas_sigmas[k],v_sigmas[k],phi_sigmas[k],psi_sigmas[k],c_sigmas[k]
            
        if d_b >=8 and d_b<9:
            k = 5
            mu_density,mu_elas,mu_v,mu_phi,mu_psi,mu_c = density_mus[k],elas_mus[k],v_mus[k],phi_mus[k],psi_mus[k],c_mus[k]
            sigma_density, sigma_elas, sigma_v, sigma_phi, sigma_psi, sigma_c = density_sigmas[k],elas_sigmas[k],v_sigmas[k],phi_sigmas[k],psi_sigmas[k],c_sigmas[k]
            
        if d_b >=9 and d_b<10:
            k = 6 
            mu_density,mu_elas,mu_v,mu_phi,mu_psi,mu_c = density_mus[k],elas_mus[k],v_mus[k],phi_mus[k],psi_mus[k],c_mus[k]
            sigma_density, sigma_elas, sigma_v, sigma_phi, sigma_psi, sigma_c = density_sigmas[k],elas_sigmas[k],v_sigmas[k],phi_sigmas[k],psi_sigmas[k],c_sigmas[k]
                
        list_Elas.append(np.random.normal(mu_elas, sigma_elas))
        list_density.append(np.random.normal(mu_density, sigma_density))
        list_v.append(np.random.normal(mu_v, sigma_v))
        list_phi.append(np.random.normal(mu_phi, sigma_phi))
        list_psi.append(np.random.normal(mu_psi, sigma_psi))
        list_c.append(np.random.normal(mu_c, sigma_c))
    
    fig, axs = plt.subplots(2, 3, figsize=(8,6))
    
    mus = elas_mus
    sigmas = elas_sigmas
    x = np.linspace(0, 100, 500)
    for i in range(7):
        mu = mus[i]
        sigma = sigmas[i]
        y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
        if i == 0:
            range_str = f'Layer {i+1} (3-4)'
        if i == 1:
            range_str = f'Layer {i+1} (4-5)'
        if i == 2:
            range_str = f'Layer {i+1} (5-6)'
        if i == 3:
            range_str = f'Layer {i+1} (6-7)'
        if i == 4:
            range_str = f'Layer {i+1} (7-8)'
        if i == 5:
            range_str = f'Layer {i+1} (8-9)'
        if i == 6:
            range_str = f'Layer {i+1} (9-10)'
        axs[0,0].plot(x, y, label=range_str)
    axs[0,0].set_title('Normal Distributions')
    axs[0,0].set_xlabel('Elastic Modulus')
    axs[0,0].set_ylabel('Probability density')
    axs[0,0].legend(title='Layers')
    
    mus = density_mus
    sigmas = density_sigmas
    x = np.linspace(2.0e-10, 3.0e-09, 500)
    for i in range(7):
        mu = mus[i]
        sigma = sigmas[i]
        y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
        if i == 0:
            range_str = f'Layer {i+1} (3-4)'
        if i == 1:
            range_str = f'Layer {i+1} (4-5)'
        if i == 2:
            range_str = f'Layer {i+1} (5-6)'
        if i == 3:
            range_str = f'Layer {i+1} (6-7)'
        if i == 4:
            range_str = f'Layer {i+1} (7-8)'
        if i == 5:
            range_str = f'Layer {i+1} (8-9)'
        if i == 6:
            range_str = f'Layer {i+1} (9-10)'
        axs[0,1].plot(x, y, label=range_str)
    axs[0,1].set_title('Normal Distributions')
    axs[0,1].set_xlabel('Density')
    axs[0,1].set_ylabel('Probability density')
    axs[0,1].legend(title='Layers')
        
    mus = v_mus
    sigmas = v_sigmas
    x = np.linspace(0.1, 0.45, 500)
    for i in range(7):
        mu = mus[i]
        sigma = sigmas[i]
        y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
        if i == 0:
            range_str = f'Layer {i+1} (3-4)'
        if i == 1:
            range_str = f'Layer {i+1} (4-5)'
        if i == 2:
            range_str = f'Layer {i+1} (5-6)'
        if i == 3:
            range_str = f'Layer {i+1} (6-7)'
        if i == 4:
            range_str = f'Layer {i+1} (7-8)'
        if i == 5:
            range_str = f'Layer {i+1} (8-9)'
        if i == 6:
            range_str = f'Layer {i+1} (9-10)'
        axs[0,2].plot(x, y, label=range_str)
    axs[0,2].set_title('Normal Distributions')
    axs[0,2].set_xlabel('v')
    axs[0,2].set_ylabel('Probability density')
    axs[0,2].legend(title='Layers')
    
    mus = phi_mus
    sigmas = phi_sigmas
    x = np.linspace(10, 50, 500)
    for i in range(7):
        mu = mus[i]
        sigma = sigmas[i]
        y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
        if i == 0:
            range_str = f'Layer {i+1} (3-4)'
        if i == 1:
            range_str = f'Layer {i+1} (4-5)'
        if i == 2:
            range_str = f'Layer {i+1} (5-6)'
        if i == 3:
            range_str = f'Layer {i+1} (6-7)'
        if i == 4:
            range_str = f'Layer {i+1} (7-8)'
        if i == 5:
            range_str = f'Layer {i+1} (8-9)'
        if i == 6:
            range_str = f'Layer {i+1} (9-10)'
        axs[1,0].plot(x, y, label=range_str)
    axs[1,0].set_title('Normal Distributions')
    axs[1,0].set_xlabel('phi')
    axs[1,0].set_ylabel('Probability density')
    axs[1,0].legend(title='Layers')
    
    mus = psi_mus
    sigmas = psi_sigmas
    x = np.linspace(0, 45, 500)
    for i in range(7):
        mu = mus[i]
        sigma = sigmas[i]
        y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
        if i == 0:
            range_str = f'Layer {i+1} (3-4)'
        if i == 1:
            range_str = f'Layer {i+1} (4-5)'
        if i == 2:
            range_str = f'Layer {i+1} (5-6)'
        if i == 3:
            range_str = f'Layer {i+1} (6-7)'
        if i == 4:
            range_str = f'Layer {i+1} (7-8)'
        if i == 5:
            range_str = f'Layer {i+1} (8-9)'
        if i == 6:
            range_str = f'Layer {i+1} (9-10)'
        axs[1,1].plot(x, y, label=range_str)
    axs[1,1].set_title('Normal Distributions')
    axs[1,1].set_xlabel('psi')
    axs[1,1].set_ylabel('Probability density')
    axs[1,1].legend(title='Layers')
    
    mus = c_mus
    sigmas = c_sigmas
    x = np.linspace(0.0004, 0.1, 500)
    for i in range(7):
        mu = mus[i]
        sigma = sigmas[i]
        y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
        if i == 0:
            range_str = f'Layer {i+1} (3-4)'
        if i == 1:
            range_str = f'Layer {i+1} (4-5)'
        if i == 2:
            range_str = f'Layer {i+1} (5-6)'
        if i == 3:
            range_str = f'Layer {i+1} (6-7)'
        if i == 4:
            range_str = f'Layer {i+1} (7-8)'
        if i == 5:
            range_str = f'Layer {i+1} (8-9)'
        if i == 6:
            range_str = f'Layer {i+1} (9-10)'
        axs[1,2].plot(x, y, label=range_str)
    axs[1,2].set_title('Normal Distributions')
    axs[1,2].set_xlabel('C')
    axs[1,2].set_ylabel('Probability density')
    axs[1,2].legend(title='Layers')
    
    fig.tight_layout()
    fig.set_size_inches(16,10)
    # Display the plot
    canvas=fig.canvas
    img_buffer = io.BytesIO()
    canvas.print_png(img_buffer)
    normal_graph = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    return depth,list_Elas,list_density,list_v,list_phi, list_psi, list_c



def Run(interval):
    global depth, list_Elas, list_density, list_v, list_phi, list_psi, list_c, predicted_graph, properties_graph, step_length, box_graph
    #d_b_list = arr[:,0]
    #E_list = arr[:,1]
    print(list_Elas)
    d_b_list = depth
    E_list = list_Elas
    density_list = list_density
    v_list = list_v
    phi_list = list_phi
    psi_list = list_psi
    c_list = list_c
    no = len(depth)
    ANN1 = 'Final_ANN1.h5'
    model1 = load_model(ANN1,custom_objects={'RSquare' : tfa.metrics.RSquare})
    scaler = pd.read_pickle('scale_ann1.pkl')
    batch = 32
        
    result = []
    para = []
    paras = []
    for i in range(no):
        #d_b,density,E,v,phi,psi,c = d_b_list[i],np.random.choice(density_list),E_list[i],np.random.choice(v_list),np.random.choice(phi_list),np.random.choice(psi_list),np.random.choice(c_list)
        d_b,density,E,v,phi,psi,c = depth[i],density_list[i],E_list[i],v_list[i],phi_list[i],psi_list[i],c_list[i]
        temp1 = [density,E,v,phi,psi,c]
        temp = [d_b,density,E,v,phi,psi,c]
        paras.append(temp1)
        para.append(temp)
        res_temp = ANN1_Predict(model1,scaler,d_b,density,E,v,phi,psi,c)
        result.append(res_temp)
        
    # Create x-axis values
    x = np.arange(no)
    para1 = np.reshape(para,(no,7))
    # Generate some random y values
    para_d_b = para1[:,0]
    para_density = para1[:,1]
    para_elastic = para1[:,2]
    para_v = para1[:,3]
    para_phi = para1[:,4]
    para_psi = para1[:,5]
    para_c = para1[:,6]

    # Create a figure with three subplots
    fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(8, 8))

    # Plot each line in a separate subplot
    axs[0].plot(x,para_d_b )
    axs[0].set_title('d_b')
    axs[1].plot(x, para_density)
    axs[1].set_title('Density')
    axs[2].plot(x, para_elastic)
    axs[2].set_title('Elastic Modulus')
    axs[3].plot(x, para_v)
    axs[3].set_title('v')
    axs[4].plot(x, para_phi)
    axs[4].set_title('phi')
    axs[5].plot(x, para_psi)
    axs[5].set_title('psi')
    axs[6].plot(x, para_c)
    axs[6].set_title('c')

    # Add x and y axis labels
    for ax in axs:
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Properties')

    # Add some space between the subplots
    fig.tight_layout()

    # Display the plot
    canvas=fig.canvas
    img_buffer = io.BytesIO()
    canvas.print_png(img_buffer)
    properties_graph = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    # Define the number of input parameters and time steps
    num_inputs = 6
    time_steps = interval
    # Generate some random data for demonstration purposes
    data = paras

    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Create the training data using a sliding window approach
    X_train = []
    y_train = []
    for i in range(time_steps, len(data)):
        X_train.append(data[i-time_steps:i, :])
        y_train.append(data[i, :])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Reshape the input data to be 3-dimensional in the form (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], time_steps, num_inputs))

    # Define the LSTM model
    model = Sequential()
    model.add(GRU(512, input_shape=(time_steps, num_inputs), return_sequences=True))
    model.add(GRU(300, activation="relu", return_sequences=True))
    model.add(GRU(200, activation="relu"))
    model.add(Dense(num_inputs))
    metric = tfa.metrics.r_square.RSquare()
    model.compile(optimizer='adam', loss='mse', metrics=[metric], run_eagerly=True)

    # Train the model
    model.fit(X_train, y_train, epochs=250, batch_size= batch)

    # Generate predictions for the next n time steps
    n = no
    last_steps = data[-time_steps:, :]
    predictions = []
    for i in range(n):
        next_step = model.predict(last_steps.reshape(1, time_steps, num_inputs))
        predictions.append(next_step)
        last_steps = np.concatenate((last_steps[1:], next_step), axis=0)

    # Reshape predictions to be 2-dimensional in the form (time steps, features)
    predictions = np.array(predictions).reshape(n, num_inputs)

    # Inverse transform the predictions back to the original scale
    predictions = scaler.inverse_transform(predictions)

    # Print the predictions
    print(predictions)

    new_db =[]
    new_elas = []
    new_density = []
    new_v = []
    new_phi = []
    new_psi =[]
    new_c = []
    for i in range(n):
        # new_db.append(predictions[i][0])
        new_density.append(predictions[i][0])
        new_elas.append(predictions[i][1])
        new_v.append(predictions[i][2])
        new_phi.append(predictions[i][3])
        new_psi.append(predictions[i][4])
        new_c.append(predictions[i][5])
        
    db =[]
    elas= []
    v = []
    phi = []
    psi = []
    c = []
    density = []
    for i in range(no):
        db.append(para[i][0])
        density.append(para[i][1])
        elas.append(para[i][2])
        v.append(para[i][3])
        phi.append(para[i][4])
        psi.append(para[i][5])
        c.append(para[i][6])

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    time = list(range(2*no))

    # Plot Elastic Modulus
    axs[0, 0].plot(time[:no], elas, label='known')
    axs[0, 0].plot(time[no:], new_elas, label='predicted', linestyle='--')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Elastic Modulus')
    axs[0, 0].legend()

    # Plot Density
    axs[0, 1].plot(time[:no], density, label='known')
    axs[0, 1].plot(time[no:], new_density, label='predicted', linestyle='--')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].legend()

    # Plot v
    axs[0, 2].plot(time[:no], v, label='known')
    axs[0, 2].plot(time[no:], new_v, label='predicted', linestyle='--')
    axs[0, 2].set_xlabel('Time Step')
    axs[0, 2].set_ylabel('v')
    axs[0, 2].legend()

    # Plot phi
    axs[1, 0].plot(time[:no], phi, label='known')
    axs[1, 0].plot(time[no:], new_phi, label='predicted', linestyle='--')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('phi')
    axs[1, 0].legend()

    # Plot psi
    axs[1, 1].plot(time[:no], psi, label='known')
    axs[1, 1].plot(time[no:], new_psi, label='predicted', linestyle='--')
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].set_ylabel('psi')
    axs[1, 1].legend()

    # Plot c
    axs[1, 2].plot(time[:no], c, label='known')
    axs[1, 2].plot(time[no:], new_c, label='predicted', linestyle='--')
    axs[1, 2].set_xlabel('Time Step')
    axs[1, 2].set_ylabel('c')
    axs[1, 2].legend()
    
    canvas=fig.canvas
    img_buffer = io.BytesIO()
    canvas.print_png(img_buffer)
    predicted_graph = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    ### Using this for saving pickle file for scaler and hdh5 file for RNN model (model)
    
    # save the scaler
    # dump(scaler, open('scale_RNN_case1', 'wb'))
    # model.save("RNN_case1.h5")
    
    ss_1 = [x for x in db[:step_length] if x >= 3 and x <4]
    ss_2 = [x for x in db[:step_length] if x >= 4 and x <5]
    ss_3 = [x for x in db[:step_length] if x >= 5 and x <6]
    ss_4 = [x for x in db[:step_length] if x >= 6 and x <7]
    ss_5 = [x for x in db[:step_length] if x >= 7 and x <8]
    ss_6 = [x for x in db[:step_length] if x >= 8 and x <9]
    ss_7 = [x for x in db[:step_length] if x >= 9 and x <=10]
    ss = [len(ss_1),len(ss_2),len(ss_3),len(ss_4),len(ss_5),len(ss_6),len(ss_7)]
    ss_layer = []
    temp = 0
    for i in ss:
        temp = temp + i
        ss_layer.append(temp)

    df_f = pd.DataFrame({'d/b':db ,'Elas': elas, 'density': density, 'phi': phi, 'psi': psi, 'v':v, 'c':c})
    df_f_i = pd.DataFrame({'d/b':db ,'Elas': new_elas,'density': new_density, 'phi': new_phi, 'psi': new_psi, 'v':new_v, 'c':new_c})

    # Create a list of parameters to plot
    parameters = ['Elas', 'density', 'v', 'phi', 'psi', 'c']
    fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(12, 24))

    # Loop through each parameter and generate a plot
    for index, var in enumerate(parameters):
        # Create example lists of data
        L1_down = list(df_f[0:ss_layer[0]][var].values)
        L2_down =  list(df_f[ss_layer[0]:ss_layer[1]][var].values)
        L3_down = list(df_f[ss_layer[1]:ss_layer[2]][var].values)
        L4_down = list(df_f[ss_layer[2]:ss_layer[3]][var].values)
        L5_down =  list(df_f[ss_layer[3]:ss_layer[4]][var].values)
        L6_down = list(df_f[ss_layer[4]:ss_layer[5]][var].values)
        L7_down = list(df_f[ss_layer[5]:ss_layer[6]][var].values)
        L7_up =  list(df_f[ss_layer[6]:ss_layer[6]+ss_layer[0]][var].values)
        L6_up = list(df_f[ss_layer[6]+ss_layer[0]:ss_layer[6]+ss_layer[1]][var].values)
        L5_up = list(df_f[ss_layer[6]+ss_layer[1]:ss_layer[6]+ss_layer[2]][var].values)
        L4_up = list(df_f[ss_layer[6]+ss_layer[2]:ss_layer[6]+ss_layer[3]][var].values)
        L3_up = list(df_f[ss_layer[6]+ss_layer[3]:ss_layer[6]+ss_layer[4]][var].values)
        L2_up = list(df_f[ss_layer[6]+ss_layer[4]:ss_layer[6]+ss_layer[5]][var].values)
        L1_up = list(df_f[ss_layer[6]+ss_layer[5]:(ss_layer[6]+ss_layer[6])-1][var].values)

        L1_down_i = list(df_f_i[0:ss_layer[0]][var].values)
        L2_down_i =  list(df_f_i[ss_layer[0]:ss_layer[1]][var].values)
        L3_down_i = list(df_f_i[ss_layer[1]:ss_layer[2]][var].values)
        L4_down_i = list(df_f_i[ss_layer[2]:ss_layer[3]][var].values)
        L5_down_i =  list(df_f_i[ss_layer[3]:ss_layer[4]][var].values)
        L6_down_i = list(df_f_i[ss_layer[4]:ss_layer[5]][var].values)
        L7_down_i = list(df_f_i[ss_layer[5]:ss_layer[6]][var].values)
        L7_up_i =  list(df_f_i[ss_layer[6]:ss_layer[6]+ss_layer[0]][var].values)
        L6_up_i = list(df_f_i[ss_layer[6]+ss_layer[0]:ss_layer[6]+ss_layer[1]][var].values)
        L5_up_i = list(df_f_i[ss_layer[6]+ss_layer[1]:ss_layer[6]+ss_layer[2]][var].values)
        L4_up_i = list(df_f_i[ss_layer[6]+ss_layer[2]:ss_layer[6]+ss_layer[3]][var].values)
        L3_up_i = list(df_f_i[ss_layer[6]+ss_layer[3]:ss_layer[6]+ss_layer[4]][var].values)
        L2_up_i = list(df_f_i[ss_layer[6]+ss_layer[4]:ss_layer[6]+ss_layer[5]][var].values)
        L1_up_i = list(df_f_i[ss_layer[6]+ss_layer[5]:(ss_layer[6]+ss_layer[6])-1][var].values)
    # Combine the data into a list for boxplot function
        data = [L1_down,L2_down,L3_down, L4_down, L5_down,L6_down, L7_down, L7_up, L6_up, L5_up, L4_up, L3_up, L2_up, L1_up, L1_down_i,L2_down_i,L3_down_i, L4_down_i, L5_down,L6_down_i, L7_down_i, L7_up_i, L6_up, L5_up_i, L4_up_i, L3_up_i, L2_up_i, L1_up_i]

        # Generate the boxplot

        col = ['blue','red']
        layer_colors = []
        for i in range(28):
            if i<14:
                layer_colors.append(col[0])
            else:
                layer_colors.append(col[1])

        # Create a legend for the plot

        # Generate the boxplot with custom box colors
        bp = axs[index].boxplot(data, boxprops=dict(facecolor='white', color='black', linewidth=2.5), patch_artist=True)

        for patch, color in zip(bp['boxes'], layer_colors):
            patch.set_facecolor(color)

        #axs.set_xticklabels(layers)
        axs[index].set_title(var)
    
    
    canvas=fig.canvas
    img_buffer = io.BytesIO()
    canvas.print_png(img_buffer)
    box_graph = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    return predicted_graph, normal_graph, box_graph

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("webapp.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    global depth, list_Elas, list_density, list_phi, list_psi, list_v, list_c, step_length
    if request.method == 'POST':
        interval = int(request.form["interval"])
        step_length = int(request.form["step_length"])
        
        elas_params = ([float(request.form["E_mean1"]), float(request.form["E_mean2"]), 
                           float(request.form["E_mean3"]), float(request.form["E_mean4"]),
                           float(request.form["E_mean5"]), float(request.form["E_mean6"]), 
                           float(request.form["E_mean7"])],
                          [float(request.form["E_sd1"]), float(request.form["E_sd2"]), 
                           float(request.form["E_sd3"]), float(request.form["E_sd4"]),
                           float(request.form["E_sd5"]), float(request.form["E_sd6"]), 
                           float(request.form["E_sd7"])])
        
        density_params = ([float(request.form["density_mean1"]), float(request.form["density_mean2"]), 
                           float(request.form["density_mean3"]), float(request.form["density_mean4"]),
                           float(request.form["density_mean5"]), float(request.form["density_mean6"]), 
                           float(request.form["density_mean7"])],
                          [float(request.form["density_sd1"]), float(request.form["density_sd2"]), 
                           float(request.form["density_sd3"]), float(request.form["density_sd4"]),
                           float(request.form["density_sd5"]), float(request.form["density_sd6"]), 
                           float(request.form["density_sd7"])])
        
        v_params = ([float(request.form["v_mean1"]), float(request.form["v_mean2"]), 
                           float(request.form["v_mean3"]), float(request.form["v_mean4"]),
                           float(request.form["v_mean5"]), float(request.form["v_mean6"]), 
                           float(request.form["v_mean7"])],
                          [float(request.form["v_sd1"]), float(request.form["v_sd2"]), 
                           float(request.form["v_sd3"]), float(request.form["v_sd4"]),
                           float(request.form["v_sd5"]), float(request.form["v_sd6"]), 
                           float(request.form["v_sd7"])])
        
        phi_params = ([float(request.form["phi_mean1"]), float(request.form["phi_mean2"]), 
                           float(request.form["phi_mean3"]), float(request.form["phi_mean4"]),
                           float(request.form["phi_mean5"]), float(request.form["phi_mean6"]), 
                           float(request.form["phi_mean7"])],
                          [float(request.form["phi_sd1"]), float(request.form["phi_sd2"]), 
                           float(request.form["phi_sd3"]), float(request.form["phi_sd4"]),
                           float(request.form["phi_sd5"]), float(request.form["phi_sd6"]), 
                           float(request.form["phi_sd7"])])
        
        psi_params = ([float(request.form["psi_mean1"]), float(request.form["psi_mean2"]), 
                           float(request.form["psi_mean3"]), float(request.form["psi_mean4"]),
                           float(request.form["psi_mean5"]), float(request.form["psi_mean6"]), 
                           float(request.form["psi_mean7"])],
                          [float(request.form["psi_sd1"]), float(request.form["psi_sd2"]), 
                           float(request.form["psi_sd3"]), float(request.form["psi_sd4"]),
                           float(request.form["psi_sd5"]), float(request.form["psi_sd6"]), 
                           float(request.form["psi_sd7"])])
        
        c_params = ([float(request.form["c_mean1"]), float(request.form["c_mean2"]), 
                           float(request.form["c_mean3"]), float(request.form["c_mean4"]),
                           float(request.form["c_mean5"]), float(request.form["c_mean6"]), 
                           float(request.form["c_mean7"])],
                          [float(request.form["c_sd1"]), float(request.form["c_sd2"]), 
                           float(request.form["c_sd3"]), float(request.form["c_sd4"]),
                           float(request.form["c_sd5"]), float(request.form["c_sd6"]), 
                           float(request.form["c_sd7"])])
        
        depth,list_Elas,list_density,list_v,list_phi, list_psi, list_c  = data_ses(step_length, elas_params, density_params,
                                                                                   v_params, phi_params, psi_params,
                                                                                   c_params)
        Run(interval)
    return render_template("webapp.html", interval=interval, step_length=step_length, ran=True, properties_graph=properties_graph,
                           predicted_graph=predicted_graph, normal_graph=normal_graph, box_graph=box_graph)
    
if __name__ =='__main__':
    app.run(debug = False)

# %%
