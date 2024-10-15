# Importando bibliotecas necessárias
import pandas as pd 
import cv2 as cv
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
import time
from modelo import classificar_imagem


#Hiperparametros
MAIN_SEED = 60
LR = 0.001
BATCH_SIZE = 32
EPOCH = 30  # Increased for better training
IMAGE_RESIZE_X = 200
IMAGE_RESIZE_Y = 200
KEEP_COLOR = False
USE_LESS_DATA = True

#SELEÇÃO DE PAGINA
st.sidebar.header('Escolha uma pagina')

escolhas = ['Pagina Inicial','Previsão de Células Cancerígenas']
escolha_do_indicadores = st.sidebar.selectbox('Selecione a página que você quer ver :',escolhas)



#CARREGANDO MODELO
model = load_model('modelo/cnn_v4.h5')

# CRIANDO FUNÇÃO DE PRE-PROCESSAMENTO DOS DADOS
def pre_process_image(image):
    if image is None:
        print("Failed to load image.")
        return None

    # Converter a imagem para escala de cinza
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Aplicar limiarização para separar o fundo e o objeto
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    
    # Aplicar a máscara para remover o fundo
    result = cv.bitwise_and(image, image, mask=thresh)
    
    # Definir as áreas pretas como brancas
    result[thresh == 0] = [255, 255, 255]
    
    # Encontrar os limites do objeto (não fundo)
    (x, y, z_) = np.where(result > 0)

    if x.size == 0 or y.size == 0:  # Verifica se a imagem está vazia
        print("Nenhuma área válida encontrada na imagem.")
        return None

    # Recortar a imagem para o objeto detectado
    mnx, mxx = np.min(x), np.max(x)
    mny, mxy = np.min(y), np.max(y)
    crop_img = image[mnx:mxx, mny:mxy, :]

    # Redimensionar a imagem para o tamanho desejado
    resized_image = cv.resize(crop_img, (IMAGE_RESIZE_X, IMAGE_RESIZE_Y))

    # Normalizar a imagem para valores entre 0 e 1
    return cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY) / 255.0  # Escala de cinza normalizada

#FUNÇÃO PARA CLASSIFICAR IMAGENS

def classificar_imagem():
    # Upload da imagem
    st.title("Sistema de Previsão de Células Cancerígenas")
    uploaded_file = st.file_uploader("Arraste ou carregue a imagem aqui:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Ler o arquivo como um array NumPy
        img_array = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv.imdecode(img_array, cv.IMREAD_COLOR)

        # Exibir a imagem carregada
        st.image(img, caption='Imagem carregada', use_column_width=True)
        
        # Preprocessar a imagem
        img_array = pre_process_image(img)  
        if img_array is not None:  # Verifica se a imagem foi processada com sucesso
            # Ajustar as dimensões da imagem para a entrada do modelo
            img_array = img_array.reshape(1, IMAGE_RESIZE_Y, IMAGE_RESIZE_X, 1)

            # Fazer a previsão com o modelo carregado
            prediction = model.predict(img_array)

            # Criar um DataFrame com as probabilidades das duas classes
            df_predictions = pd.DataFrame({
                'Probabilidade_0': 1 - prediction[:, 0],  # probabilidade da classe 0
                'Probabilidade_1': prediction[:, 0]       # probabilidade da classe 1
            })

            # Obter a classe com maior probabilidade
            class_idx = np.argmax(prediction)

            # Exibir resultados
            probabilidade_1 = df_predictions['Probabilidade_1'].iloc[0]*100  

            #CRIANDO FAIXAS DE PROBABILIDADE
            if probabilidade_1 <=20:
                st.title('Baixíssima Probabilidade')
                st.subheader(f'Há uma probabilidade de {probabilidade_1:.2f}% de que está célula seja cancerígena')
                st.write("A análise realizada pelo sistema indica uma probabilidade extremamente baixa de que a célula seja cancerígena. Apesar do resultado, recomendamos que o paciente continue com exames regulares conforme as orientações médicas e os protocolos do laboratório.")
            elif probabilidade_1 <=40:
                st.title('Baixa probabilidade ')
                st.subheader(f'Há uma probabilidade de {probabilidade_1:.2f}% de que está célula seja cancerígena')
                st.write("O sistema detectou uma baixa probabilidade de malignidade na célula. Embora o risco seja reduzido, é aconselhável que o paciente mantenha o acompanhamento periódico e siga as recomendações do laboratório para monitoramento contínuo.")
            elif probabilidade_1 <=60:
                st.title('Média Probabilidade (ATENÇÂO)')
                st.subheader(f'Há uma probabilidade de {probabilidade_1:.2f}% de que está célula seja cancerígena')
                st.write("Atenção: O sistema identificou uma probabilidade moderada de que a célula seja cancerígena. Recomendamos que o paciente seja encaminhado a um especialista para realizar exames complementares e garantir uma avaliação mais aprofundada, conforme os protocolos do laboratório.")
            elif probabilidade_1 <=80:
                st.title('Alta Probabilidade')
                st.subheader(f'Há uma probabilidade de {probabilidade_1:.2f}% de que está célula seja cancerígena')
                st.write("O sistema identificou uma alta probabilidade de que a célula seja cancerígena. Sugerimos que o paciente seja imediatamente encaminhado para uma avaliação com um especialista, a fim de confirmar o diagnóstico e, se necessário, iniciar um tratamento apropriado.")
            else:
                st.title('Altíssima Probabilidade')
                st.subheader(f'Há uma probabilidade de {probabilidade_1:.2f}% de que está célula seja cancerígena')
                st.write("ALERTA: O sistema aponta uma probabilidade muito alta de que a célula seja cancerígena. É crucial que o paciente seja encaminhado com urgência para uma consulta com um especialista para diagnóstico e intervenção imediata, conforme os protocolos de tratamento recomendados pelo laboratório.")





def homepage():
    st.title('Radeon Labs - Inovação que salva vidas!')
    st.write('''Bem-vindos ao Radeon Labs, um centro de excelência em pesquisa e inovação no uso de machine learning para a saúde. 
    Nosso foco é desenvolver soluções avançadas para prever a probabilidade de um paciente possuir células cancerígenas, promovendo diagnósticos mais rápidos e precisos.''')
    st.image('imagens/LOGO PROJETO DS.png')





#Condicionais para seleção de pagina
 
if escolha_do_indicadores == 'Pagina Inicial':
    #PAGINA INICIAL 
    homepage()

elif escolha_do_indicadores == 'Previsão de Células Cancerígenas':
    #PAGINA DE IMPLEMENTAÇÃO MODELO
    classfication = classificar_imagem()




        







