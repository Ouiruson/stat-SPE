# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'monospace','sans-serif':['Helvetica'],\
         'weight' : 'normal','size' : 15}
#font = {'weight' : 'demibold','size' : 15}
plt.rc('font', **font)
#plt.rc('text', usetex=True)
#plt.rc('text.latex',unicode=True)
# fig = {'dpi': 300,'edgecolor': 'w','format':'eps'}
fig = {'dpi': 300,'edgecolor': 'w'}
plt.rc('savefig',**fig)

#plt.rc(block=False)

# ==============================================================================
# Ce module contient des définitions de fonctions python destinées à la 
# présentation graphique de données. 
# Dernière mise à jour: 15/09/2016
# ==============================================================================

# ==============================================================================
# Les fonctions de la bibliothèque plot_lib sont les suivantes:
# boxplot:      Tracé de diagrammes boîtes (à moustache)
# mosaicplot    Représentation graphique d'un tableau de contingence
# pieplot :     Camembert
# plotcp :      Tracé des projections des variables sur plan des 
#               composantes principales cp
# plotpf :      Tracé des projections des individus sur plan des axes principaux 
# qqplot :      QQ-plot of two vectors or of a vector and a normal distribution
# scatterplotS: Scatter plot de deux vecteurs de tailles identiques
# ==============================================================================

## ==============================================================================
#function boxplot(X,labX,varargin)
## ==============================================================================
  #""" Diagramme en boite d'un vecteur colonne X
      #Second argument: label pour chaque colonne
      #Les arguments suivants (optionnels) sont identiques aux deux premiers, 
      #i.e. des couples Xi, labXi afin de tracer plusieurs boites. """
## ==============================================================================

#nin = argn(2);   # nombre d'arguments dans la commande d'appel
#nX = nin/2;      # nombre de boites à tracer

#for k=1:nX,  # détermination des min, max, médiane et quantiles de X
  #if k > 1,
     #X = varargin(2*k-3);    # si arguments optionnels, X est ... 
                             ## l'argument d'ordre impair
  #Xmax(k) = max(X(:));       
  #Xmin(k) = min(X(:)); 
  #X50(k) = median(X(:));
  #Xtab = tabul(X(:),'i');    # «help tabul»
  #Xquant = cumsum(Xtab(:,2))/sum(Xtab(:,2));  
  #i25 = min(find(Xquant >= 0.25));
  #i75 = min(find(Xquant >= 0.75));
  #X25(k) = Xtab(i25,1);
  #X75(k) = Xtab(i75,1);


#for k=1:length(Xmin),
  #x0 = 2*k-1;
  #plot([x0-0.5 x0+0.5],[X25(k) X25(k)],'k')
  #plot([x0-0.5 x0+0.5],[X75(k) X75(k)],'k')
  #plot([x0-0.5 x0-0.5],[X25(k) X75(k)],'k')
  #plot([x0+0.5 x0+0.5],[X25(k) X75(k)],'k')
  #plot([x0 x0],[Xmin(k) X25(k)],'k')
  #plot([x0-0.25 x0+0.25],[Xmin(k) Xmin(k)],'k')
  #plot([x0 x0],[X75(k) Xmax(k)],'k')
  #plot([x0-0.25 x0+0.25],[Xmax(k) Xmax(k)],'k')
  #plot([x0-0.5 x0+0.5],[X50(k) X50(k)],'k',"thickness",2)

  #if k == 1,
    #a = gca();
    #a.font_size = 2;
  #else
    #labX = varargin(2*k-2);
  
  #a.x_ticks.locations(k) = 2*k-1;
  #a.x_ticks.labels(k) = labX;
  
  #if k == length(Xmin),
    #dX = 0.05*(max(Xmax) - min(Xmin));
    #a.data_bounds = [0 min(Xmin)-dX; 2*k max(Xmax) + dX];
    #a.tight_limits = "on";
  

#a.x_ticks.locations(k+1:$) = 2*k+1;
#a.x_ticks.labels(k+1:$) = '';

## ==============================================================================

## ==============================================================================
#function XC = centrage(X)
  #""" centrage:  fonction de centrage par colonne  XC = (X-Moy)
      #ou X sont les donnees (en colonne), Moy leurs moyennes. """
## ==============================================================================

#UN = ones(size(X,1),1);
#Moy = mean(X,1);   # moyenne de chaque colonnes => vecteur ligne
#Moy = UN * Moy;
## Calcul de la matrice XN des données centrées-réduites
#XC  = (X - Moy);

## ==============================================================================

## ==============================================================================
#function XN = centrage_reduction(X)
  #""" centrage_reduction : normalisation des  colonnes par centrage
      #et reduction avec la formule XN = (X-Moy) / Std
      #ou X sont les donnees, Moy leurs moyennes, Std leurw ecarts types. """
## ==============================================================================

#UN = ones(size(X,1),1);
#Moy = mean(X,1);   # moyenne des colonnes => vecteur ligne
#Moy = UN * Moy;
#Std = stdev(X,1);
#Std = UN * Std;
## Calcul de la matrice XN des données centrées-réduites
#XN  = (X - Moy)./Std;

## ==============================================================================

## ==============================================================================
#function rho = corrcoef(X,Y);
## ==============================================================================
## Calcul du coeff de corrélation linéaire entre deux vecteurs 
## de même taille
## ==============================================================================

#s = size(X);
#if and(s > 1), error("corrcoef: X n''est pas un vecteur"), 
#s = size(Y);
#if and(s > 1), error("corrcoef: Y n''est pas un vecteur"), 

#lX = size(X(:),1);
#lY = size(Y(:),1);

#if lX ~= lY, error("corrcoef: X n''est pas de meme taille que Y"), 

#rho = correl(X(:),Y(:),eye(lX,lX));


## ==============================================================================

## ==============================================================================
#function [X, Ilab, Vlab] = lec_tab2_tsv(fich,nl,nc,substitute)
## ==============================================================================
## lecture d'un tableau 2D dans un fichier tsv
## La premiere ligne décrit le nom des variables 
##    et la premiere colonne le nom des individus.
## En entrée: fich : le nom de fichier (chaine de car), 
##            nl   : le nombre d'individus (-1 si inconnu)
##            nc   : le nombre de variables.
##            si substitute = [',' '.']: substitution ',' par '.'
##            si pas de substitution à faire, omettre cet argument 
## ==============================================================================

#if ~isfile(fich), 
   #error("Nom de fichier non valide"),


#nin = argn(2);  # nombre d'arguments d'entrée
#if nin == 3, substitute = [];   # pas de substutution

#fid    = mopen(fich);
#lignes = stripblanks(mgetl(fid));
#mclose(fid);

#if ~isempty(substitute),  # substitution (usuellement des ',' par des '.')
   #lignes= strsubst(lignes,substitute(1),substitute(2));


## séparateurs: Tab ou blancs? 
#ind  = strindex(lignes(1),ascii(9));
#if ~isempty(ind),
   #sep = ascii(9);
#else
   #sep = ' ';


#Vlab = strsplit(lignes(1),sep);
#Vlab = Vlab($-nc+1:$);

#if nl <=0, nl = size(lignes,1), 

#for i = 2:nl,
   #li = strsplit(lignes(i),sep);
   #ni = size(li,1);
   #n1 = ni - nc + 1;
   #ii = min(strindex(lignes(i),li(n1)));
   #Ilab(i) = part(lignes(i),1:ii-1);
   #li = li(n1:$);
   #kn =find(stripblanks(li) == ''); ; li(kn) = string(%nan);
   #X(i,:) = evstr(li(:))';



## ==============================================================================

## ==============================================================================
#function [X, Ilab, Vlab] = lec_tab2_csv(fich,nl,nc,sep,substitute)
## ==============================================================================
## lecture d'un tableau 2D dans un fichier .csv
## La premiere ligne décrit le nom des variables 
##    et la premiere colonne le nom des individus.
## En entrée: fich : le nom de fichier (chaine de car), 
##            nl   : le nombre d'individus (-1 si inconnu)
##            nc   : le nombre de variables.
##            si substitute = [',' '.']: substitution ',' par '.'
##            si pas de substitution à faire, omettre cet argument 
## ==============================================================================

##printf("sep = %s\n",sep)

#if ~isfile(fich), 
  #error("Nom de fichier non valide"),


#nargin = argn(2);

#if nargin < 4 | isempty(sep),
  #sep = ",";


#if nargin < 5,
   #substitute = [];


#fid    = mopen(fich);
#lignes = stripblanks(mgetl(fid));
#mclose(fid);

#if ~isempty(substitute),  # substitution (usuellement des ',' par des '.')
   #lignes= strsubst(lignes,substitute(1),substitute(2));


#Vlab = strsplit(lignes(1),sep);
#Vlab = Vlab($-nc+1:$);

#if nl <=0, nl = size(lignes,1), 

#for i = 2:nl,
   #li = strsplit(lignes(i),sep);
   #ni = size(li,1);
   #Ilab(i-1) = li(1);
   #li = li($-nc+1:$);
   #kn = find(stripblanks(li) == ''); li(kn) = string(%nan);
   #X(i-1,:) = evstr(li(:))';


#Ilab = strsubst(Ilab,sep,'');


## ==============================================================================

## ==============================================================================
#function mosaicplot(X,xlab,ylab);
## ==============================================================================
## Représentation graphique d'un tableau de contingence
## X est une matrice (n,p), effectif ou fréquence
## xlab, ylab sont des tableaux de chaine de caractères de dimension
## p et n respectivement. 
## ==============================================================================
## X = [0.11 0.23 ; 0.54 0.12]; xlab = ['H' 'F']; ylab = ['S' 'D']
## X = [Hv Fv; H-Hv F-Fv];xlab=['H' 'F'];ylab=['D' 'S'];

#[nl nc] = size(X);

#soml = sum(X,1),
#somc = sum(X,2);

#a = gca();
#x(1) = 0; y = ones(nl,nc);
#rect = [];

#w = soml/sum(soml);

#plot2d(0,0,0,"010"," ",[-0.05,-0.05,1,1]) # définit les coordonnées utilisateur

#for j=1:nc,
 #for i = 1:nl,
   #h(i,j) = X(i,j)/soml(j);
   #rect = [x(j) y(i,j) w(j)-0.01 h(i,j)-0.01]
   #y(i+1,j) = y(i,j) - h(i,j);
   #xrect(rect)
 
 #x(j+1) = x(j) + w(j);


#for j=1:nc,
  #xs = 0.5*(x(j)+x(j+1));
  #ys = -0.05;
  #xstring(xs,ys,xlab(j))
  #e=gce();
  #e.font_size = 3;


#for i=nl:-1:1,
  #ys = 0.5*(y(i,1)+y(i+1,1));
  #xs = -0.05;
  #xstring(xs,ys,ylab(i));
  #e=gce();
  #e.font_size = 3;



## ==============================================================================

## ==============================================================================
#function pieplot(Xocc,Xclass)  
## Camembert
## Xocc est le nombre d'occurence dans les classes Xclass.
## Le camembert contient length(Xocc) parts. 

## Xocc = occO3; Xclass = classO3
#socc = sum(Xocc);

#for i=1:length(Xclass)-1;
  #labX(i) = string(Xclass(i)) + ' - ' + string(Xclass(i+1));
  #labX(i) = labX(i) + sprintf(" (%.1f %%)",Xocc(i)/socc*100) 


#pie(Xocc,labX)


## ==============================================================================

## ==============================================================================
#function plotcp(X,cp,labels,norme)
## ==============================================================================
## Fonction plotcp: tracé des projections des variables sur plan des 
## composantes principales cp
## X : matrice des données (variables en colonnes)
## cp : deux composantes principales en colonne. La longueur de chaque 
##      cp est égale au nombre de lignes de X (nombre d'individus)
## labels : labels des variables.
## norme : variable booléenne. Vecteur des variables normalisé ou non. 
##         Si norme = %T, <X_k,vi> = rho(X_k,vi) (rho = corrélation)
## ==============================================================================
## X = Xc; cp = [v1 v2];

#if argn(2) < 4, norme = %F; 

#x1 = X'*cp(:,1);
#x2 = X'*cp(:,2);

#if norme, 
  #mx = 1;
  #for j=1:size(X,2),
    #xn = norm(X(:,j),2);
    #x1(j) = x1(j)/xn;
    #x2(j) = x2(j)/xn;
  
#else  
  #stx = stdev(X,1)*sqrt(size(X,1)-1);
  #mx = max(stx);

#Ox1 = [zeros(x1(:)') ; x1(:)'];
#Ox2 = [zeros(x2(:)') ; x2(:)'];

#plot2d([-1.1*mx 1.1*mx], [-1.1*mx 1.1*mx], [-1,-1], "022")
#xarc(-mx,mx,2*mx,2*mx,0,360*64)
#xarrows(Ox1,Ox2,0.5)

#a = gca();
#dx = (a.data_bounds(2)-a.data_bounds(1))/100;
#dy = (a.data_bounds(4)-a.data_bounds(3))/100;
#xset("font size",3),
#for i=1:length(x1),
  #xstring(x1(i)+dx,x2(i)-dy,labels(i))

#a.font_size = 2;
#plot([a.data_bounds(1) a.data_bounds(2)],[0 0],'k'),
#plot([0 0],[a.data_bounds(3) a.data_bounds(4)],'k'),
#a.isoview = "on";

## ==============================================================================

## ==============================================================================
#function plotpf(X,ei,labels,symb)
## ==============================================================================
## Fonction plotpf: tracé des projections des individus sur plan des 
## axes principaux ui
## X : matrice des données (variables en colonnes)
## ui : deux axes principaux en colonne. La longueur de chaque 
##      axe est égale au nombre de colonnes de X (nombre de variables)
## labels : labels des individus.
## symp (optionnel) : type de symbole et couleur à tracer 
## ==============================================================================

#if argn(2) == 3, symb = 'ob'; 

#x1 = X*ei(:,1);
#x2 = X*ei(:,2);
#plot(x1,x2,symb)
#a = gca();
#dx = 0*(a.data_bounds(2)-a.data_bounds(1))/100;
#xset("font size",4),
#for i=1:length(x1),
   #xstring(x1(i)+dx,x2(i),labels(i))

#plot([a.data_bounds(1) a.data_bounds(2)],[0 0],'k'),
#plot([0 0],[a.data_bounds(3) a.data_bounds(4)],'k'),
#a.data_bounds(2) = a.data_bounds(2) + 0.07*(a.data_bounds(2)-a.data_bounds(1));


## ==============================================================================

## ==============================================================================
#function q=quantile(x,p,method)
##=======================================
## Empirical quantile
##
## Calling Sequence
## q=quantile(x,p)
## q=quantile(x,p,method)
##
## Parameters
## x : a n-by-1 matrix of doubles
## p : a m-by-1 matrix of doubles, the probabilities
## method : a 1-by-1 matrix of doubles, available values are method=1,2,3 (default=1)
## q : a m-by-1 matrix of doubles, the quantiles. q(i)is greater 
## than p(i) percents of the values in x

## Description
## The empirical quantile of the sample x, a value
## that is greater than p percent of the values in x
## If input x is a matrix then the quantile is
## computed for every column.
## If p is a vector then q is a matrix, each line contain
## the quantiles computed for a value of p.
##
## The empirical quantile is computed by one of three ways
## determined by a third input argument (with default 1).
##
## method=1. Interpolation so that F(X_(k)) == (k-0.5)/n.
## 
## method=2. Interpolation so that F(X_(k)) == k/(n+1).
##
## method=3. Based on the empirical distribution.
##
## Examples
## x=[
## 0.4827129 0.3431706 -0.4127328 0.3843994
## -0.7107495 -0.2547306 0.0290803 0.1386087
## -0.7698385 1.0743628 1.0945652 0.4365680
## -0.5913411 -0.7426987 1.609719 0.8079680
## -2.1700554 -0.7361261 0.0069708 1.4626386
## ];
## # Make a column vector:
## x=x(:);
## p=linspace(0.1,0.9,10)';
## q=quantile(x,p) # Same as : q=quantile(x,p,1)
## # Check the property
## p(1)
## length(find(x<q(1)))/length(x)
## p(5)
## length(find(x<q(5)))/length(x)
## q=quantile(x,p,2)
## q=quantile(x,p,3)
## ==========================================================

#q=[];
#[nargout,nargin] = argn(0)

#if nargin<3 then
  #method = 1;

#if min(size(x))==1 then
  #x = x(:);
  #%v1=size(p)
  #q = zeros(%v1(1),%v1(2));
#else
  #q = zeros(size(p,'*'),size(x,2));

#if min(size(p))>1 then
  #error('Not matrix p input');

#if or(p>1|p<0) then
  #error('Input p is not probability');


#%v = x
#if min(size(%v))==1 then
  #%v=gsort(%v)
#else
  #%v=gsort(%v,'r')

#x = %v($:-1:1,:);
#p = p(:);
#n = size(x,1);
#if method==3 then
  #i=ceil(min(max(1,p*n),n))
  #qq = x(i)
#else
  #x = [x(1,:);x;x(n,:)];
  #if method==2 then # This method is from Hjort's Computer
                    ## intensive statistical methods page 102
    #i = p*(n+1)+1;
  #else  # Method 1
    #i = p*n+1.5;
  
  #iu = ceil(i);
  #il = floor(i);
  #d = (i-il)*ones(1,size(x,2));
  #qq = x(il,:) .* (1-d)+x(iu,:) .* d;

#q(:) = qq;

## ==============================================================================

##=====================================================================
#function qqplot(x,a1,a2)
##=======================================================
## Create a QQ-plot of two vectors or of a vector and a normal distribution
##
## Calling Sequence
## qqplot(x), 
## qqplot(x,ps)
## qqplot(x,distrib,ps)
## qqplot(x,y)
## qqplot(x,y,ps)
##
## Parameters
## x : a n-by-1 matrix of doubles
## y : a n-by-1 matrix of doubles
## ps : a 1-by-1 matrix of strings, the plot symbol (default="*")
## distrib : a matrix of string describing the theoretical distribution
##           to compare with.
##
## Description
## If one sample is normaly distributed, it is approximatively on a 
## straight line.
##
## If two distributions are the same (or possibly linearly
## transformed) the points should form an approximately straight
## line.
##
## A red solid line joining the first and third quartiles of x and y
## is plot.
## This is a linear fit of the order statistics of the two samples.
## An extrapolation of this line below the first and over the third
## quantile is performed, and plot as a red dotted line.
##=======================================================

#[nargout,nargin] = argn(0)

## nargin = 2; a1 = 'o'; a2 = [];
## nargin = 3; a1 = y; a2 = 'r+';

#ps = '*'; y = []; distrib = "nor";

#if nargin == 2, 
  #if type(a1) == 10 then # a1 est une chaine de char
    #if length(a1) > 2, distrib = a1; else, ps = a1; 
  #else
    #y = a1;
    

#if nargin == 3;
  #if type(a1) == 10 then
    #distrib = a1; 
  #else
    #y = a1; 
  
  #ps = a2;


#%v = x(:);  
#x = gsort(%v,'r','i');
#nx = length(x);
#Fx = ((1:nx)-0.375)/(nx+0.25);  # fonction de répartition des x
#titre = "Q-Q plot "

#if ~isempty(y),
  #%v = y(:)
  #y = gsort(%v,'r','i');
  #ny = length(%v);
  #Fy = ((1:ny)-0.375)/(ny+0.25);  # fonction de répartition des y
  #xx = x; yy = y;
  #if nx ~= ny,
    #n = max(nx,ny);
    #if nx < n, xx = interp1(Fx,x,Fy); 
    #if ny < n, yy = interp1(Fy,y,Fx); 
    
  #plot(xx(:),yy(:),ps);
  #p=linspace(0.01,0.99,100)';
  #qx=quantile(xx,p);
  #qy=quantile(yy,p);
  #plot([qx(5),qx(95)],[qy(5),qy(95)],"r-")
  #titre = titre + " Y vs X";
  #title(titre,"fontsize",3);
  #xlabel('X Data Quantiles')
  #ylabel('Y Data Quantiles');
#else
  #n = length(x);
  #mu = mean(x); 
  #sigma = stdev(x);
  #Fx = ((1:n)-0.375)/(n+0.25);
  #if distrib == "nor",
    #y = mu + sqrt(2)*sigma*erfinv(2*Fx(:)-1);
    #titre = titre + "x vs Normal Distribution";
  #elseif distrib == "unf";
    #a = min(x); b = max(x); dx = (b-a)/n; a = a-dx/2; b = b + dx/2;
    #y = a + (b-a)*Fx(:);
    #ylab = "Uniform distribution Quantiles"
    #titre = titre + ": X vs Uniform Distribution";
  
  #plot(x,y,ps);
  #plot(y,y,'--r')
  #title(titre,"fontsize",3);
  #xlabel('Data Quantiles')
  #ylabel('Theoretical Quantiles');


## Add the 0.25-0.75 quantile line
## a=(q2(25)-q2(75))/(q1(25)-q1(75))
## b=q2(25)-a*q1(25)
## y1=a*x(1)+b
## plot([x(1),q1(25)],[y1,q2(25)],"r-.")
## y2=a*x($)+b
## plot([q1(75),x($)],[q2(75),y2],"r-.")

#xgrid

## ==============================================================================

## ==============================================================================
#function beta = reglin_mult(Y,X)
## ==============================================================================
## Régression multiple
## ==============================================================================

#beta = pinv(X'*X)*X'*Y;


## ==============================================================================

## ==============================================================================
#function a = scatterplot(X,Y,varargin);
## ==============================================================================
## scatter plot de deux vecteurs de tailles identiques
## le troisième argument est optionnel. Il contient un descriptif 
## de marque et/ou de couleur (ex: "r+" pour tracer des croix rouges,
## voir "help plot")
## ==============================================================================

#if argn(2) > 2,  # nombre d'arguments en entrée > 2 ?
  #symb = varargin(1);
#else
  #symb = '+k'; 


#plot(X,Y,symb)

#a = gca();
#a.font_size = 2;
#rho = corrcoef(X,Y);
#xgrid

#xmin = a.data_bounds(1);
#xmax = a.data_bounds(2);
#ymin = a.data_bounds(3);
#ymax = a.data_bounds(4);
#str = sprintf(" = %.2f",rho);
#xstring(xmin + 0.05*(xmax-xmin),ymin + 0.9*(ymax-ymin),["$\rho$" str]);
#e = gce();
#e.font_size = 4;


## ==============================================================================

# ==============================================================================
def hist_stem(X, freq = True, linefmt='b-',markerfmt='bd',\
              xlab="bins",ylab="",title="stem plot",file_name=""):
#def hist_stem(X):
   #""" histogramme en baton d'un vecteur X """
#-------------------------------------------------------------------------------

   bins,nb = np.unique(X,return_counts=True);

   if freq:
      nt = np.sum(nb)
      nb  = nb/nt; 
      ylab = "Frequency"
   else:
      ylab = "Nb of occurrences"

   print(ylab)

   plt.stem(bins,nb,linefmt = 'k-',markerfmt = 'kd'),
   dX = np.diff(np.sort(X))
   ii = np.where(dX > 0)
   deltaX = np.min(dX[ii])
   plt.xlim(bins[0]-deltaX,bins[-1]+deltaX)

   ii = np.where(nb > 0)
   deltaY = np.min(nb[ii])
   plt.ylim(plt.ylim()[0],np.max(nb)+deltaY)

   plt.grid()

   plt.xlabel("bins")
   plt.ylabel(ylab)
   plt.title("Stem plot")

   plt.show(block=False)

   if len(file_name) > 0:
      plt.savefig(file_name)
   #return bins

# ==============================================================================

## ==============================================================================
#function [pval,cimean,cistd] = test1b(x,c,b)
##==================================================
## Bootstrap test that mean equals zero
##
## Calling Sequence
## pval=test1b(x)
## pval=test1b(x,c)
## pval=test1b(x,c,b)
## [pval,cimean]=test1b(...)
## [pval,cimean,cistd]=test1b(...)
##
## Parameters
## x : a m-by-n matrix of doubles
## c : a 1-by-1 matrix of doubles, c in [0,1], the confidence level for
##     the confidence intervals (default=0.95)
## b : a 1-by-1 matrix of doubles, b>=1, the number of bootstrap samples
##     (default=2000)
## pval : a 1-by-1 matrix of doubles, the probability that the mean is zero
## cimean : a 1-by-3 matrix of doubles, the confidence interval for the mean
## cistd : a 1-by-3 matrix of doubles, the confidence interval for the 
##         standard deviation
##
## Description
## Performs the bootstrap t test for the equality of the mean to zero
## and computes confidence interval for the mean.
##
## Another name for the bootstrap t is studentized bootstrap.
##
## The confidence intervals are of the form
## [LeftLimit, PointEstimate, RightLimit]
##
## Examples
## x=grand(10,1,'nor',0,1);
## pval=test1b(x) # pval is close to 1
##
## x=distfun_normrnd(10,1,20,1);
## pval=test1b(x) # pval is close to 0
##
## x=distfun_chi2rnd(3,20,1);
## pval=test1b(x)
## # Set the confidence level
## pval=test1b(x,0.9)
## # Set the number of bootstrap samples
## pval=test1b(x,[],100)
## # Get a confidence interval for the mean
## [pval,cimean]=test1b(x)
## # Get a confidence interval for the standard
## # deviation
## [pval,cimean,cistd]=test1b(x)
##
## Authors
## Copyright (C) 2013 - Michael Baudin
## Copyright (C) 2010 - DIGITEO - Michael Baudin
## Copyright (C) 1993 - 1995 - Anders Holtsberg
##==================================================

#pval=[];
#cimean=[];
#cistd=[];
#[nargout,nargin] = argn(0)
##
#x = x(:);
#if nargin<2 then
  #c = 0.95;

#if nargin<3 then
  #b = 2000;


#n = size(x,'*')
#m = sum(x)/n;
#s = stdev(x);

#xB = zeros(n,b);
#U=grand(n*b,1,'unf',0,1)
#J = ceil(U*n);
#xB(:) = x(J);
#mB = mean(xB);
#sB = stdev(xB,'r');
#z = (mB-m) ./ sB;
#t = quantile(z,[(1-c)/2,1-(1-c)/2]);
#cimean = [m-t(2)*s,m,m-t(1)*s];
##
#tt = m/s;
#if tt>0 then
  #pval = 2*sum((mB-tt*sB)>=m)/b;
#else
  #pval = 2*sum((mB-tt*sB)<=m)/b;


#if nargout>2 then
  #d = quantile(sB/s,[(1-c)/2,1-(1-c)/2]);
  #cistd = [s/d(2),s,s/d(1)];


## ==============================================================================
