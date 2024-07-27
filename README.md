# # ¿Cuál es la mejor tarifa?
# 
# Trabajas como analista para el operador de telecomunicaciones Megaline. La empresa ofrece a sus clientes dos tarifas de prepago, Surf y Ultimate. El departamento comercial quiere saber cuál de las tarifas genera más ingresos para poder ajustar el presupuesto de publicidad.
# 
# Vas a realizar un análisis preliminar de las tarifas basado en una selección de clientes relativamente pequeña. Tendrás los datos de 500 clientes de Megaline: quiénes son los clientes, de dónde son, qué tarifa usan, así como la cantidad de llamadas que hicieron y los mensajes de texto que enviaron en 2018. Tu trabajo es analizar el comportamiento de los clientes y determinar qué tarifa de prepago genera más ingresos.

 
# ## Inicialización

# In[1]:


from scipy import stats as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# ## Cargar datos

# In[2]:


# Carga los archivos de datos en diferentes DataFrames

calls = pd.read_csv('/datasets/megaline_calls.csv')
internet = pd.read_csv('/datasets/megaline_internet.csv')
messages = pd.read_csv('/datasets/megaline_messages.csv')
plans = pd.read_csv('/datasets/megaline_plans.csv')
users = pd.read_csv('/datasets/megaline_users.csv')


# ## Preparar los datos

# ## Tarifas


plans['mb_per_month_included'] = plans['mb_per_month_included']/1024
plans = plans.rename(columns={'mb_per_month_included': 'gb_per_month_included'})
plans

# ## Usuarios/as

# In[6]:


# Imprime la información general/resumida sobre el DataFrame de usuarios
print(users.describe())
print()
print(users.info())


# In[7]:


# Imprime una muestra de datos para usuarios

print(users.sample(20))
print()
print('Hay', users.duplicated().sum(), 'filas duplicadas')
print()
print(users.isna().sum())



# Hay 8 columnas, 2 con datos tipo entero (user_id y age) y las 6 restantes como object. 
# No hay filas duplicadas, y la columna churn_date tiene 466 filas don datos ausentes (debido a que el usuario
# seguía haciendo uso del servicio).
# 
# Es necesario convertir el tipo de dato de las columnas 'reg_date' y 'churn_date' de object a datetime.
# 
# Los nombres de los títulos cumplen con las especificaciones, excepto la columna 'plan', no especifica a
# qué se refiere.
# 
# De los 500 usuarios, las edades van de los 18 a los 75 años, el promedio es de 45 años. 



# ### Corregir los datos


# In[8]:


users['reg_date']=pd.to_datetime(users['reg_date'])
users['churn_date']=pd.to_datetime(users['churn_date'])
users = users.rename(columns={'plan':'plan_name'})

users.info()


# ### Enriquecer los datos

# [Agrega factores adicionales a los datos si crees que pudieran ser útiles.]

# In[ ]:


# ## Llamadas

# In[9]:


# Imprime la información general/resumida sobre el DataFrame de las llamadas
print(calls.info())
print()
print(calls.describe())


# In[10]:


# Imprime una muestra de datos para las llamadas

print(calls.tail(20))
print()
print(calls.isnull().sum())
print()
print('Hay',calls.duplicated().sum(),'filas duplicadas')



# El dataset está compuesto por 4 columnas, 2 de tipo object (id y call_date), 1 de tipo entero (user_id)
# y uno de tipo flotante (duration). 
# Es necesario convertir los datos de la columna 'call_date' de object a datetime.
# 
# No hay filas duplicadas ni datos ausentes. Los nombres de los títulos cumplen con las especificaciones, 
# pero los títulos 'id' y 'duration' no son específicos.
# 
# De los 137735 registros de llamadas, la duración promedio es de casi 6 minutos y medio. El máximo es de más
# de media hora.
# 
# De 137735 registros de llamada, 26834 tienen una duración de 0 segundos. El 19.5% (casi la quita parte del total) son llamadas en las que probablemente ha habido un problema de conexión

# ### Corregir los datos

# 

# In[11]:


calls['call_date']=pd.to_datetime(calls['call_date'])
calls.rename(columns={'id':'calling_id', 'duration':'calling_duration'},inplace=True)
valores_cero = calls.loc[calls['calling_duration'] == 0]
print(calls)
print()
print(valores_cero)


# ### Enriquecer los datos

# 

# In[12]:


calls['calling_duration'] = calls['calling_duration']
calls['calling_duration'] = np.ceil(calls['calling_duration']).astype(int)
calls['month'] = calls['call_date'].dt.month
calls


# ## Mensajes

# In[13]:


# Imprime la información general/resumida sobre el DataFrame de los mensajes
messages.info()


# In[14]:


# Imprime una muestra de datos para los mensajes
print(messages.sample(50))
print()
print(messages.isna().sum())
print()
print('Hay',messages.duplicated().sum(),'filas duplicadas.')


# El dataset se compone de tres columnas, 2 del tipo object (user_id y message_date) y 1 de tipo 
# entero (user_id). Es necesario convertir el tipo de datos de la columna 'message_date' de object a
# datetime.
# 
# No hay filas duplicadas ni valores ausentes. Los nombres de los títulos cumplen con las especificaciones,
# excepto el título 'id' que no es específico.


# ### Corregir los datos

# 

# In[15]:


messages['message_date']=pd.to_datetime(messages['message_date'])
messages.rename(columns={'id':'message_id'},inplace=True)
print(messages)
messages.info()


# ### Enriquecer los datos

# [Agrega factores adicionales a los datos si crees que pudieran ser útiles.]

# In[16]:


messages['month'] = messages['message_date'].dt.month
messages


# ## Internet

# In[17]:


# Imprime la información general/resumida sobre el DataFrame de internet
print(internet.info())
print()
print(internet.describe())


# In[18]:


# Imprime una muestra de datos para el tráfico de internet

print(internet.sample(30))
print()
print(internet.isna().sum())
print()
print('Hay',internet.duplicated().sum(),'filas duplicadas.')



# Hay 4 columnas y tres tipos de datos distintos, 2 tipo object (id y session_date), 1 tipo entero (user_id)
# y 1 tipo flotante (mb_used). Es necesario convertir el tipo de dato de la columna 'session_date' de object
# a datetime.
# 
# El formato de los títulos cumple con el formato, pero el título 'id' no es específico.
# No hay filas duplicadas ni valores ausentes.
# 
# Se ha usado un promedio de casi 367 mb por sesión. 


# ### Corregir los datos


# In[19]:


internet['session_date']=pd.to_datetime(internet['session_date'])
internet.rename(columns={'id':'session_id'},inplace=True)

internet.info()


# ### Enriquecer los datos

# 

# In[20]:


internet['month'] = internet['session_date'].dt.month
internet['mb_used'] = internet['mb_used']/1024
internet['mb_used'] = np.ceil(internet['mb_used']).astype(int)
internet = internet.rename(columns={'mb_used': 'gb_used'})
internet



# ## Estudiar las condiciones de las tarifas

# In[21]:


# Imprime las condiciones de la tarifa y asegúrate de que te quedan claras
plans


# In[22]:


# Calcula el número de llamadas hechas por cada usuario al mes. Guarda el resultado.
monthly_calls_per_user = calls.groupby(['month','user_id']).agg({'calling_id': 'count', 'calling_duration': 'sum'}).reset_index()
monthly_calls_per_user= monthly_calls_per_user.rename(columns={'calling_id': 'calling_amount', 'calling_duration': 'minutes_per_user'})
monthly_calls_per_user


# In[23]:


# Calcula la cantidad de minutos usados por cada usuario al mes. Guarda el resultado.
monthly_calls_per_user


# In[24]:


# Calcula el número de mensajes enviados por cada usuario al mes. Guarda el resultado.
monthly_mes_per_user = messages.groupby(['month', 'user_id'])['message_id'].count().reset_index()
monthly_mes_per_user = monthly_mes_per_user.rename(columns={'message_id': 'messages_per_user'})
monthly_mes_per_user = monthly_mes_per_user.sort_values(by='month')
nuevo_orden_columnas = ['month', 'user_id', 'messages_per_user']
monthly_mes_per_userr=monthly_mes_per_user[nuevo_orden_columnas]
monthly_mes_per_user


# In[25]:


# Calcula el volumen del tráfico de Internet usado por cada usuario al mes. Guarda el resultado.
internet['gb_used'] = internet['gb_used'].apply(lambda x: np.ceil(x * 100) / 100)
internet_gb = internet.groupby(['month', 'user_id'])['gb_used'].sum().reset_index()
internet_gb


# In[26]:


# Fusiona los datos de llamadas, minutos, mensajes e Internet con base en user_id y month
consumo_general = monthly_calls_per_user.merge(monthly_mes_per_user, on=['user_id','month'], how='outer')
consumo_general = consumo_general.merge(internet_gb,on=['user_id','month'], how='outer')
consumo_general.head(10)




# In[27]:


consumo_general['messages_per_user'] = consumo_general['messages_per_user'].fillna(0)
consumo_general


# In[28]:


# Añade la información de la tarifa
tarifas = consumo_general.merge(users,on='user_id',how='right')
tarifas=tarifas.sort_values(by='month', ascending=True)
tarifas_1 = tarifas.merge(plans,on='plan_name',how='left')
tarifas_1 = tarifas_1.dropna(subset=['month'])
tarifas_1['month'] = tarifas_1['month'].astype('int64')
tarifas_1 = tarifas_1.fillna(0)
tarifas_1


# In[29]:


tarifas_1['minutes_per_user'] =tarifas_1['minutes_per_user'].round().astype(int)
print(tarifas_1.info())
print()
mean_gb_usados = tarifas_1.groupby('plan_name')['gb_used'].mean()
print(mean_gb_usados)


# 

# In[30]:


# Calcula el ingreso mensual para cada usuario    
def calcular_ingresos_mensuales(row):
    ingresos_llamadas = max(row['minutes_per_user'] - row['minutes_included'], 0) * row['usd_per_minute']
    ingresos_mensajes = max(row['messages_per_user'] - row['messages_included'], 0) * row['usd_per_message']
    ingresos_datos = max(row['gb_used'] - row['gb_per_month_included'], 0) * row['usd_per_gb']
    ingresos_totales = ingresos_llamadas + ingresos_mensajes + ingresos_datos
    ingresos_totales += row['usd_monthly_pay']
    
    return ingresos_totales

tarifas_1['ingresos_mensuales'] = tarifas_1.apply(calcular_ingresos_mensuales, axis=1)
print(tarifas_1.head(20))
print()
print(tarifas_1.info())


# ## Estudia el comportamiento de usuario

# 

# ### Llamadas

# In[31]:


# Compara la duración promedio de llamadas por cada plan y por cada mes. Traza un gráfico de barras para visualizarla.
mean_calls = tarifas_1.groupby(['month','plan_name'])['minutes_per_user'].mean().reset_index()
mean_calls['month_plan'] = mean_calls['month'].astype(str) + '_' + mean_calls['plan_name']
mean_calls.plot(title='Promedio de minutos por plan y por mes', x='month_plan', 
                y='minutes_per_user', xlabel = 'mes/plan', ylabel='promedio de minutos',
               figsize=[10,10], kind= 'bar')
print(mean_calls)
print()
plt.show()




# In[32]:


# Compara el número de minutos mensuales que necesitan los usuarios de cada plan. Traza un histograma.
sum_minutes_calls = tarifas_1.groupby(['month','plan_name'])['minutes_per_user'].sum()
sum_minutes_calls.plot(title='Total de minutos por plan y por mes', x=['month','plan_name'], 
                y='minutes_per_user', xlabel = 'mes/plan', ylabel='Total de minutos',
               figsize=[10,10], kind= 'bar')

print(sum_minutes_calls)
print()


# 

# In[33]:


# Calcula la media y la varianza de la duración mensual de llamadas.
stats_call_duration=tarifas_1.groupby(['month','plan_name'])['minutes_per_user'].agg(['mean', 'var']).reset_index()
print(stats_call_duration)
print()
tarifas_1['minutes_per_user'].hist(bins=100)
plt.show()



# In[34]:


# Traza un diagrama de caja para visualizar la distribución de la duración mensual de llamadas
plt.boxplot(sum_minutes_calls)


# Sí. Los usuarios que tienen el plan 'Ultimate' hacen más uso de las llamadas que aquellos que tienen el
# plan 'Surf', pero también influye el mes.


# ### Mensajes

# In[35]:


# Comprara el número de mensajes que tienden a enviar cada mes los usuarios de cada plan
sum_monthly_msgs = tarifas_1.groupby(['month','plan_name'])['messages_per_user'].sum()
print(sum_monthly_msgs)
print()
tarifas_1['messages_per_user'].hist(bins=100, figsize=[10,10])
plt.show()


# In[36]:


sum_monthly_msgs = tarifas_1.groupby(['month','plan_name'])['messages_per_user'].sum().reset_index()
mensajes_por_plan = sum_monthly_msgs.groupby('plan_name')['messages_per_user'].sum()
mensajes_por_plan


# In[37]:


# Compara la cantidad de tráfico de Internet consumido por usuarios por plan
internet_used = tarifas_1.groupby(['month','plan_name','user_id']).agg({'gb_used':'mean'}).reset_index()
print(internet_used)
print()
internet_used['gb_used'].hist(bins=100)
plt.show()


# In[38]:


mean_gb_per_plan = internet_used.groupby('plan_name')['gb_used'].mean()
print(internet_used)
print()
print(mean_gb_per_plan)
print()
print(mean_gb_per_plan.describe())
print()
plt.boxplot(internet_used['gb_used'])
plt.show()

# La cantidad de mensajes enviados por plan es:
# plan_name
# surf        40.140496
# ultimate    41.080556
# 
# Los usuarios del plan Surf tienden a enviar casi la misma cantidad de mensajes que los usuarios del con 
# el plan Ultimate.

# ### Internet

# In[41]:


print(internet_used)
print()
internet_used['gb_used'].hist(bins=100, figsize=[10,10])
plt.show()
print()
plt.boxplot(internet_used['gb_used'])
plt.show()


# In[53]:


gb_used_month_plan = internet_used.groupby(['month', 'plan_name'])['gb_used'].sum()
gb_used_month_plan.plot(title='Total de gb por plan y por mes', x=['month','plan_name'], 
                y='gb_used', xlabel = 'mes/plan', ylabel='Total de gb',
               figsize=[10,10], kind= 'bar')
plt.show()



# In[42]:


mean_gb_per_plan = internet_used.groupby('plan_name')['gb_used'].mean()
print(mean_gb_per_plan)
print()
print(internet_used.describe())



# Promedio mensual de gb usador por usuario por plan es de:
# surf        40.140496
# ultimate    41.080556
# 
# El consumo es casi igual, independientemente del plan. El promedio de uso es de casi 40 gb.

# 

# ## Ingreso

# 


# In[56]:


monthly_incomes = tarifas_1.groupby(['month', 'plan_name']).agg({'ingresos_mensuales': 'sum'}).reset_index()
monthly_incomes['month_plan'] = monthly_incomes['month'].astype(str) + '_' + monthly_incomes['plan_name']
monthly_incomes.plot(title='Ingresos por plan y por mes', x='month_plan', 
                y='ingresos_mensuales', xlabel = 'mes/plan', ylabel='Total de ingresos',
               figsize=[10,10], kind= 'bar')

print(monthly_incomes)
print()
plt.show()
print()
plt.boxplot(monthly_incomes['ingresos_mensuales'])
plt.show()
print()
monthly_incomes['ingresos_mensuales'].hist(bins=23)
plt.show()




# In[45]:


ingreso_anual = monthly_incomes.groupby('plan_name')['ingresos_mensuales'].sum().reset_index()
porcentaje_surf = (ingreso_anual.iloc[0,1] / ingreso_anual['ingresos_mensuales'].sum())*100
porcentaje_ultimate = (ingreso_anual.iloc[1,1] / ingreso_anual['ingresos_mensuales'].sum())*100

print(ingreso_anual)
print()
print('El plan surf aporta el', porcentaje_surf,'% del ingreso total anual.')
print('El plan ultimate aporta el', porcentaje_ultimate,'% del ingreso total anual.')


# In[46]:


surf = tarifas_1[tarifas_1['plan_name'] == 'surf']['ingresos_mensuales'].describe()
print('Los datos estadísticos del plan Surf son:')
print(surf)
print()
ultimate = tarifas_1[tarifas_1['plan_name'] == 'ultimate']['ingresos_mensuales'].describe()
print('Los datos estadísticos del plan Ultimate son:')
print(ultimate)



# El plan surf representó alrededor del 79% de los ingresos anuales, mientras que el plan ultimate  
# representó el 21% de los ingresos anuales. 
# 
# En promedio, el ingreso  mensual del plan Surf es de USD$282.12, mientras que el del plan Ultimate es de
# USD$166.63.
# 

# ## Prueba las hipótesis estadísticas

# Hipótesis nula 1:
#     El ingreso promedio de los usuarios de las tarifas Ultimate y Surf difiere.
# Hipótesis alternativa 1:
#     El ingreso promedio de los usiarios de ambas tarifas es igual.
# 
#     
#     
#     
# Hipótesis nula 2:    
#     El ingreso promedio de los usuarios en el área de Nueva York-Nueva Jersey es diferente al de 
#     los     usuarios de otras regiones.
# Hipótesis alternativa 2:
#     El ingreso promedio de los usuarios del área de Nueva York - Nueva Jersey es igual al de los usuarios
#     de otras regiones.


# In[47]:


surf= tarifas_1[tarifas_1['plan_name'] == 'surf']['ingresos_mensuales'].var()
surf_muestra= tarifas_1[tarifas_1['plan_name'] == 'surf']['ingresos_mensuales'].count()
print('La varianza del plan Surf es:',surf)
print('El tamaño de la muestra es de', surf_muestra, 'usuarios.')
print()
ultimate = tarifas_1[tarifas_1['plan_name'] == 'ultimate']['ingresos_mensuales'].var()
ultimate_muestra = tarifas_1[tarifas_1['plan_name'] == 'ultimate']['ingresos_mensuales'].count()
print('La varianza del plan Ultimate es:', ultimate)
print('El tamaño de la muestra es de', ultimate_muestra, 'usuarios.')




# In[48]:


from scipy.stats import levene
surf= tarifas_1[tarifas_1['plan_name'] == 'surf']['ingresos_mensuales']
ultimate = tarifas_1[tarifas_1['plan_name'] == 'ultimate']['ingresos_mensuales']
statistic, p_value = levene(surf, ultimate)

print("Estadístico de la prueba de Levene:", statistic)
print("Valor p:", p_value)

alfa = 0.05
if p_value < alfa:
    print("Hay evidencia para rechazar la hipótesis nula de igualdad de varianzas.")
else:
    print("No hay suficiente evidencia para rechazar la hipótesis nula de igualdad de varianzas.")


# In[49]:


# Prueba las hipótesis
ingresos_surf = tarifas_1.loc[tarifas_1['plan_name'] == 'surf', 'ingresos_mensuales'] 
ingresos_ultimate = tarifas_1.loc[tarifas_1['plan_name'] == 'ultimate', 'ingresos_mensuales'] 

surf_mean = ingresos_surf.mean()
alpha = 0.5

results_plan = st.ttest_ind(ingresos_surf, ingresos_ultimate, equal_var=False)
print('valor p: ',results_plan.pvalue)

if (results_plan.pvalue < alpha): # compara el valor p con el umbral alpha
    print('Rechazamos la hipótesis nula')
else:
    print("No podemos rechazar la hipótesis nula")



# In[50]:


# Prueba las hipótesis
ingresos_ny_nj = tarifas_1.loc[tarifas_1['city'] == 'New York-Newark-Jersey City, NY-NJ-PA MSA', 'ingresos_mensuales'] 
ingresos_others = tarifas_1.loc[tarifas_1['city'] != 'New York-Newark-Jersey City, NY-NJ-PA MSA', 'ingresos_mensuales']
surf_mean = ingresos_surf.mean()
alpha = 0.5

results_region = st.ttest_ind(ingresos_ny_nj, ingresos_others, equal_var=False)
print('valor p: ',results_region.pvalue)

if (results_region.pvalue < alpha): # compara el valor p con el umbral alpha
    print('Rechazamos la hipótesis nula')
else:
    print("No podemos rechazar la hipótesis nula")


# ## Conclusión general
# A pesar de que el cobro se hace en gb, los registros de actividad de los usuarios se mide en mb, lo que implica un retrabajo
# al momento de analizar la información, ya que es necesario convertir las cantidades. 
# 
# Durante el 2018, el plan surf representó alrededor del 40% de los ingresos anuales, mientras que el plan ultimate representó 
# casi el 60% de los ingresos anuales, siendo el número de usuarios (en la muestra) casi el doble de los usuarios del plan 
# Ultimate.
# 
# Ningún usuario consumió el total de los gb incluidos en su respectivo plan, y pocos usuarios rebasaron el límite de minutos 
# y de mensajes incluídos. El promedio de mensajes, minutos y gb usados está por debajo del límite del plan contratado. Además, 
# el consumo en ambos planes es muy parecido, independientemente de la región.
