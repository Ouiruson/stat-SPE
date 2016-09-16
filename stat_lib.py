# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.stats as sps

# ==============================================================================
# Ce module contient des définitions de fonctions python destinées au traitement 
# statistiques des données, essentiellement des fonctions pour la présentation 
# graphique de données. 
# Dernière mise à jour: 12/09/2016
# ==============================================================================

# ==============================================================================
# Les fonctions de la bibliothèque stat_lib sont les suivantes:
# centrage:     Centrage d'un tableau par colonne
# centrage_reduction: Centrage-réduction  d'un tableau par colonne
# corrcoeff:    Calcul du coeff de corrélation linéaire entre 
#               deux vecteurs de même taille
# quantile :    Empirical quantile
# reglin_mult:  Régression multiple
# test1b:       Bootstrap test that mean equals zero
# ==============================================================================

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
