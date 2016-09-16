#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import matplotlib.pyplot as mpl
from   importlib import reload
#------------------------------
import modules.plot_lib as pl
import modules.stat_lib as sl
import modules.io_lib   as io 
#-----------------------------

dir_figs = "../plots/"

#======================================================================
# Le fichier "stat_lib.py" contient une collection de fonctions 
#    python dédiées au traitement statistique des données. 
# Le fichier plot_lib.py contient des fonctions graphiques
# Le fichier io_lib.py contient des fonctions dédiées à la lecture/écriture
#    de données.
# Ces fichiers sont dans un réperoire "modules"
#
# Jetez y un  coup d'oeil au contenu de ces fichiers afin de voir 
# les fonctions existantes 
#======================================================================

# Données body.dat (http://www.amstat.org/publications/jse/datasets/)

# lecture du tableau "body.dat" comportant 25 colonnes
# Le fichier "body.dat" se trouve dans le répertoire ../data/

data = np.loadtxt("../data/body.dat");  # tableau de 507 lignes, 25 colonnes 

# on range les colonnes qui nous intéressent dans des vecteurs colonnes
# ayant des noms explicites
poids  = data[:,22];  
taille = data[:,23]; 
sexe   = data[:,24];  # 0: féminin; 1/ masculin
age    = data[:,21];

# Diagramme baton  ====================================================
mpl.figure(1) 
mpl.clf() 
# hist_stem: une fonction graphique de la bibliothèque "plot_lib"
b_freq = False
if b_freq:
   ylb = "Frequency"
else:
   ylb = "Nb of occurrences"

titre   = "Diagramme baton des tailles" # titre
nom_fig = dir_figs + "dbaton_tailles.pdf"

pl.hist_stem(taille,freq=b_freq,xlab="Taille (cm)",ylab=ylb,title=titre,\
             file_name=nom_fig);  

xlabel();               # label de l'abscisse
savefig();  # Export de la figure 1 dans un fichier 
                                         # pdf, le fichier est sauvé dans le 
                                         # répertoire courant.

## Histogramme  ========================================================
## A faire: Définir le vecteur abscisse x avec un pas de 2.5 cm 
## (opérateur ':' ou fonction "linspace")
#scf(2); clf, 
#hist(x,taille)
#xlabel("Taille (cm)");
#ylabel("Frequence")
#title("Histogramme des tailles")
#savefig("../plots/hist_tailles.pdf");

## Effectifs/Fréquences cummulé(e)s  ===================================
## A faire: Créer le tableu Ts contenant les tailles triées en ordre 
## croissant, Ts. Calculer les fréquences cumulées correspondantes, F_T, 
## à l'aide des fonctions "tabul", "cumsum" et "sum".

#scf(3); clf,
#plot2d2(Ts,F_T),
#xlabel("Tailles (cm)")
#ylabel("Fréquences cummulées")
#title("Diagramme cummulatif des tailles")
#a = gca(); 
#a.box = "on";
## export de la figure dans un fichier pdf nommé "Fcum_tailles.pdf"
#xs2pdf(3,"Fcum_tailles.pdf");

## Diagramme boite  ====================================================
## A faire: Calculer Tmin,T25,T50,T75,Tmax ?

#scf(4); clf,
#boxplot(Tmin,T25,T50,T75,Tmax,"Taille")
#title("Diagramme boite des tailles")
#ylabel("Tailles (cm)")
#xs2pdf(4,"dboite_tailles.pdf");

## Nuages de points  ===================================================
## A faire: tracer le scatter plot poids-taille
##          Calculer la droite de régression (fonction reglin)
##          Calculer le coef de correlation poids-taille (fonction correl)

