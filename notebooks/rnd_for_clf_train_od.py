import pandas as pd
import numpy as np

data = pd.read_csv('../DATA/final_train_od_dummies.csv')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(np.array(data['Outcome']).ravel())
data['Outcome'] = le.transform(np.array(data['Outcome']).ravel())

X = data[['Tax Related', 'Number of Lawyers',
     'Number of Legal Parties', 'Value formatted',
       'Unified Contribution formatted', 'Milano', 'Bari', 'Bologna', 'Genova',
       'Palermo', 'Napoli', 'Torino', 'Trento', 'Roma', "L'Aquila", 'Potenza',
       'Perugia', 'Campobasso', 'Firenze', 'Cagliari', 'Venezia', 'Cosenza',
       'Ancona', 'Trieste', 'Aosta','OR-140999', 'OR-145009', 'OR-139999',
       'OR-145999', 'OR-130099', 'OR-101003', 'OR-130121', 'OR-130111',
       'OR-130131', 'OR-101002', 'OR-180002', 'OSA-180002','OSA-180099','OSA-180001','OSA-140999','OSA-145999']]

y = data[['Duration', 'Outcome']]

X_to_scale = X[['Number of Lawyers','Number of Legal Parties', 'Value formatted',
       'Unified Contribution formatted']]
X_not_to_scale = X[['Tax Related','Milano', 'Bari', 'Bologna', 'Genova',
       'Palermo', 'Napoli', 'Torino', 'Trento', 'Roma', "L'Aquila", 'Potenza',
       'Perugia', 'Campobasso', 'Firenze', 'Cagliari', 'Venezia', 'Cosenza',
       'Ancona', 'Trieste', 'Aosta','OR-140999', 'OR-145009', 'OR-139999',
       'OR-145999', 'OR-130099', 'OR-101003', 'OR-130121', 'OR-130111',
       'OR-130131', 'OR-101002', 'OR-180002', 'OSA-180002','OSA-180099','OSA-180001','OSA-140999','OSA-145999']]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

y = pd.DataFrame(y)
std_scale = StandardScaler()
X_scaled = std_scale.fit_transform(X_to_scale)

X_scaled_df = pd.DataFrame(X_scaled, columns=[X_to_scale.columns])
X_scaled_df = pd.concat([X_scaled_df, X_not_to_scale], axis=1)

X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled_df, y, random_state=0,
                                                            test_size=0.1)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,random_state=0,
                                                            test_size=0.2  )

X_train_val = pd.DataFrame(np.array(X_train_val), columns=[['Number of Lawyers','Number of Legal Parties', 
       'Value formatted','Unified Contribution formatted', 'Tax Related','Milano', 'Bari', 'Bologna', 'Genova',
       'Palermo', 'Napoli', 'Torino', 'Trento', 'Roma', "L'Aquila", 'Potenza',
       'Perugia', 'Campobasso', 'Firenze', 'Cagliari', 'Venezia', 'Cosenza',
       'Ancona', 'Trieste', 'Aosta','OR-140999', 'OR-145009', 'OR-139999',
       'OR-145999', 'OR-130099', 'OR-101003', 'OR-130121', 'OR-130111',
       'OR-130131', 'OR-101002', 'OR-180002', 'OSA-180002',
       'OSA-180099','OSA-180001','OSA-140999','OSA-145999']], index = X_train_val.index)


X_test = pd.DataFrame(np.array(X_test), columns=[['Number of Lawyers','Number of Legal Parties', 
       'Value formatted','Unified Contribution formatted', 'Tax Related','Milano', 'Bari', 'Bologna', 'Genova',
       'Palermo', 'Napoli', 'Torino', 'Trento', 'Roma', "L'Aquila", 'Potenza',
       'Perugia', 'Campobasso', 'Firenze', 'Cagliari', 'Venezia', 'Cosenza',
       'Ancona', 'Trieste', 'Aosta','OR-140999', 'OR-145009', 'OR-139999',
       'OR-145999', 'OR-130099', 'OR-101003', 'OR-130121', 'OR-130111',
       'OR-130131', 'OR-101002', 'OR-180002', 'OSA-180002',
       'OSA-180099','OSA-180001','OSA-140999','OSA-145999']], index = X_test.index)

y_train_val_duration = y_train_val['Duration']
y_train_val_Outcome = y_train_val['Outcome']
y_test_duration = y_test['Duration']
y_test_Outcome = y_test['Outcome']
y_train_duration = y_train['Duration']
y_train_Outcome = y_train['Outcome']
y_val_duration = y_val['Duration']
y_val_Outcome = y_val['Outcome']
y_train_val_duration = pd.DataFrame(y_train_val_duration)
y_train_val_Outcome = pd.DataFrame(y_train_val_Outcome)
y_test_duration = pd.DataFrame(y_test_duration)
y_test_Outcome = pd.DataFrame(y_test_Outcome)
y_train_duration = pd.DataFrame(y_train_duration)
y_train_Outcome = pd.DataFrame(y_train_Outcome )
y_val_duration = pd.DataFrame(y_val_duration)
y_val_Outcome = pd.DataFrame(y_val_Outcome)

import itertools as it
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
# import shutil

param_grid = {'max_depth':np.arange(1,8),
               'min_samples_split':np.arange(5,100,10),
               'min_samples_leaf':np.arange(5,100,10),
               'n_estimators':np.arange(1,500,50)}


allNames = sorted(param_grid)
combinations = it.product(*(param_grid[Name] for Name in allNames))
combinations = list(combinations)

df = pd.DataFrame(combinations, columns=param_grid.keys())

start_point = int(input('START POINT:'))
end_point = int(input('END POINT:'))


#shutil.rmtree('/content/drive/MyDrive/RFC')
#os.makedirs('/content/drive/MyDrive/RFC')

# FROM 4596 TO 5000
best_f1 = np.load('../RFC/'+os.listdir('../RFC')[-1],allow_pickle=True)[1] 
for i in range(start_point,end_point):
  clf = RandomForestClassifier(max_depth = df.iloc[i][0],
                               min_samples_split = df.iloc[i][1],
                               min_samples_leaf = df.iloc[i][2],
                               n_estimators = df.iloc[i][3],
                               class_weight = 'balanced',
                               n_jobs=1)
  clf.fit(X_train_val, np.array(y_train_val_Outcome).ravel())
  clf_val = np.array([clf.get_params(), f1_score(np.array(y_test_Outcome).ravel(),clf.predict(X_test), average = 'weighted')])
  if clf_val[1]>best_f1:
    best_f1 = clf_val[1]
    #np.save(os.path.join('/content/drive/MyDrive/RFC/', 'best_up_to_' + str(i)), clf_val)
    np.save(os.path.join('../RFC', 'best_up_to_'+str(i)),clf_val)
  else:
    # os.rename('/content/drive/MyDrive/RFC/' + os.listdir('/content/drive/MyDrive/RFC/')[-1],
    #           os.path.join('/content/drive/MyDrive/RFC/', 'best_up_to_' + str(i) + '.npy'))
    os.rename('../RFC/'+os.listdir('../RFC')[-1], os.path.join('../RFC', 'best_up_to_'+str(i)+'.npy'))
    


