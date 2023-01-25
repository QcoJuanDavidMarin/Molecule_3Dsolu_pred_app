# Bibliografia https://huggingface.co/spaces/osanseviero/SMILES_RDKit_Py3DMOL_FORK/blob/main/app.py
# https://www.youtube.com/watch?v=kMYrWUCvAfE
# https://birdlet.github.io/2019/10/02/py3dmol_example/

import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import py3Dmol
import streamlit.components.v1 as components
import cirpy # Convertir de cas a SMILES
from cirpy import Molecule # Para extraer caracteristicas de las moleculas

# Configuración de la página
#PAGE_CONFIG = {"page_title":"JuanDavidMarin", "page_icon":":smiley","layout":"centered"}
#st.set_page_config(PAGE_CONFIG)

st.set_page_config(page_title='JuanDavidMarin', layout='wide')
### Cunstom Function

def AromaticProportion(m):
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count = []
    for i in aromatic_atoms:
        if i==True:
            aa_count.append(1)
    AromaticAtom = sum(aa_count)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    AR = AromaticAtom/HeavyAtom
    return AR

def generate(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)


    i=0
    for mol in moldata:

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["MolLogP","MolWt","NumRotatableBonds","AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors


######################
# Page Title
######################



st.write('''
# Predictor de solubilidad molecular
## Juan David Marín
''')
#components.html("<p style='color:red;'>Juan David Marín </p>")

st.sidebar.header('Ingreso Estructura')

SMILES_input = 'NCCCC'

SMILES = st.sidebar.text_area('Ingrese una Fórmula', SMILES_input)
SMILES = 'C\n' + SMILES #Adds C as a dummy, first item
SMILES = SMILES.split('\n')

st.header('Compute Molecular descriptors')
X = generate(SMILES)
st.table(X[1:])# Skips the dummy first item

mol_list = []
for i in SMILES[1:]:
    mol = Chem.MolFromSmiles(i)
    mol = Draw.MolToImage(mol, size= (300,130))
    mol_list.append(mol)
st.sidebar.image(mol_list)


### Graficos en 3D
def show(smi, style='stick'):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    #AllChem.MMFFOptimizeMolecule(mol, maxIters = 200)
    mblock = Chem.MolToMolBlock(mol)

    view = py3Dmol.view(width=600, height=600)
    view.addModel(mblock, 'mol')
    view.setStyle({style:{}})
    view.addSurface('VDW', {'opacity':0.5, 'colorscheme':{'gradient':'rwb'}})
    view.zoomTo()
    view.show()
    view.render()
    t =view.js()
    f = open('viz.html', 'w')
    f.write(t.startjs)
    f.write(t.endjs)
    f.close()


compound_smiles = SMILES[1]
m = Chem.MolFromSmiles(compound_smiles)
Draw.MolToFile(m,'mol.png')
show(compound_smiles)
HtmlFile = open("viz.html", 'r', encoding='utf-8')
source_code1 = HtmlFile.read()
c1,c2= st.columns(2)
with c1:
    components.html(source_code1, height = 500,width=500)
with  c2:
    with st.expander('Nombres de la molecula'):
        st.write(cirpy.resolve(compound_smiles, 'names'))


# Cargar el modelo y hacer las predicciones
load_model = pickle.load(open(r'D:\Python Scripts\Proyectos\Stramlit\Solubility app\solubility_model.pkl', 'rb'))
prediction = load_model.predict(X[1:])
st.title('Predicción solubilidad (Log S)')
st.metric(label='Predicción', value=prediction, 
delta='Coeficientes: -0.74145216 -0.00648553 -0.00508588 -0.51922781',
delta_color="inverse" )
