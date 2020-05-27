clc; close all; clear all;
x = csvread('dims_new.csv');

y = kmo(x)

[ndim,prob,chisquare] = barttest(x)

cronbach(x)