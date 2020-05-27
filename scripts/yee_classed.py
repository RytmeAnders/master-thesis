import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
import math
import itertools
from scipy import stats
import statistics as st
import Levenshtein as l
import seaborn as sns
from pylab import *

plt.style.use('seaborn') # pretty matplotlib plots

class Yee:
    def __init__(self, factor, d):
        self.labels = ['action','social','mastery','achievement','immersion','creativity']
        self.d = d
        
        # Weights
        self.wn = 1/6 # Weight normal
        self.wb = self.wn * factor # Weight biased
        self.wub = (1 - self.wb) / 5 # Weight unbiased
        self.weights  = [[self.wb, self.wub, self.wub, self.wub, self.wub, self.wub],
                         [self.wub, self.wb, self.wub, self.wub, self.wub, self.wub],
                         [self.wub, self.wub, self.wb, self.wub, self.wub, self.wub],
                         [self.wub, self.wub, self.wub, self.wb, self.wub, self.wub],
                         [self.wub, self.wub, self.wub, self.wub, self.wb, self.wub],
                         [self.wub, self.wub, self.wub, self.wub, self.wub, self.wb]]
    
    def run(self):
        """Runs all the appropriate operations"""
        # Calculate unbiased and biased types implicitly
        self.d['pt'] = [self.calculate_type(x,y,False) for (x,y) in zip(self.d.type, self.d.bias)]
        self.d['pt_biased'] = [self.calculate_type(x,y,True) for (x,y) in zip(self.d.type, self.d.bias)]
        
        self.describe()
        
        print("Biased weight: " + str(self.wb))
        print("The rest: " + str(self.wub))
        
    def set_factor(self, factor):
        """Set the weight of the bias"""
        self.factor = factor
    
    def print_nums(self, raw, unbias, bias):
        """Prints the numbers associated with the player type calculation.
        Used for debugging"""
        print("Raw numbers: " + str(raw))
        print("Unbiased type: " + str(unbias))
        print("Biased type: " + str(bias))
        
    def type_correlation(self, i, j):
        """Check how similar two arrays are. Eg. used to check how much the
        bias factors into the calculation"""
        x = 0
        
        for k in range(len(i)):
            if i[k] == j[k]:
                x = x + 1
        
        return x/len(i)
    
    def calculate_strength(self, d, weighted):
        """Calculates the strength of your player type by looking at the
        ratio between your type score and the other scores.
        d = the calculated player type scores (1x6 vector).
        weighted = the biased array is an ndarray, so it needs to be cast
        to a list first"""
        if weighted:
            d = list(d)

        t = d[np.argmax(d)]
        d.pop(np.argmax(d))
        dmean = st.mean(d)
        
        return t/dmean
        
    
    def describe(self):        
        """Plots descriptive statistics of the dataset"""
        # Count values
        age_count = self.d['age'].value_counts(sort=True)
        gender_count = self.d.gender.value_counts(sort=False)
        experience_count = self.d['experience'].value_counts(sort=False)
        
        pt_count = self.d['pt'].value_counts()
        pt_biased = self.d['pt_biased'].value_counts()
        
        fig, axs = plt.subplots(2,2, figsize=(20,12))       
        # age
        axs[0,0].bar(age_count.index,age_count)
        axs[0,0].set_ylim(0,25)
        axs[0,0].set_title('Age',fontsize=16)
        axs[0,0].set_ylabel('Frequency',fontsize=16)
        
        #Gender
        axs[0,1].bar(gender_count.index,gender_count)
        axs[0,1].set_ylim(0,40)
        axs[0,1].set_title('Gender',fontsize=16)
        axs[0,1].set_ylabel('Frequency',fontsize=16)
        
        #Experience
        axs[1,0].bar(experience_count.index,experience_count)
        axs[1,0].set_ylim(0,25)
        axs[1,0].set_title('Experience',fontsize=16)
        axs[1,0].set_ylabel('Frequency',fontsize=16)
        
        #Player type
        axs[1,1].bar(pt_biased.index,pt_biased)
        axs[1,1].set_ylim(0,25)
        axs[1,1].set_title('Player Types',fontsize=16)
        axs[1,1].set_ylabel('Frequency',fontsize=16)
            
        fig.suptitle("Descriptive statistics of choice test (N={})"
                     .format(len(self.d)), fontsize=30)
        
        plt.show()
        
    def calculate_type(self, pt, bias, weighted):
        """Calculate a players player type, either with or without bias.
        pt = string of questionnaire answers.
        bias =  their preferred player type.
        weighted = bool if the player type should be calculated with bias"""
        # Convert int to int array
        t = [int(d) for d in str(pt)]
        
        #Get every second index
        list1 = [x for x in t[::2]]
        list2 = [x for x in t[1::2]]
        
        # Get the pairwise mean per index between the lists
        g = [(x+y)/2 for x,y in zip(list1,list2)] 
    
        # Calculate unbiased player types
        g_unbiased = [x*self.wn for x in g]  #Weight the player types
        ptc = self.labels[np.argmax(g_unbiased)]
        
        # Calculate biased player types
        for j in range(len(self.labels)):
            if self.labels[j] == bias:
                g_biased = np.multiply(g, self.weights[j])      
        ptbc = self.labels[np.argmax(g_biased)]
        
        # Print the numbers (for debugging!)
        #self.print_nums(g, g_unbiased, g_biased)
        
        uscore = self.calculate_strength(g_unbiased, False)
        bscore = self.calculate_strength(g_biased, True)
        ratio = bscore/uscore
        #print("Unbiased: " + str(uscore))
        #print("Biased: " + str(bscore))
        #print("Ratio: " + str(ratio))
        
        # Return unbiased and biased playertypes
        if weighted:
            return ptbc
        return ptc
    
    def cluster(self):
        """Groups players according to type.
        Row indices in the output correspond to label indices"""
        x = []
        
        for label in self.labels:
            x.append(self.d.loc[self.d.pt_biased == label].reset_index())
            
        return x
    
    def distances(self, inputs):
        """Calculates edit distances within a dataset.
        Inputs = The array of raw data."""
        x = []
        
        for t in inputs:
            x.append([l.distance(x,y) for (x,y) in list(itertools.combinations(t.path,2))])
            
        return x
    
    def is_parametric(self, d):
        """Checks if a dataset is parametric."""
        sig = []
        
        # Check for normality
        normal = [stats.shapiro(x) for x in d]
        pvalues = [i[1] for i in normal]
        sig = [x for x in pvalues if x<0.05]
        
        # Check for homogeneity of variance
        eqvar = [stats.levene(x,y) for (x,y) in list(itertools.combinations(d,2))]
        sig.append([x.pvalue for x in eqvar if x.pvalue<0.05])
        
        # If sig is empty (if there are no significant values), it is parametric
        if sig == []:
            return True
        return False
    
    def test(self, d, posthoc):
        """Performs the relevant inferential statistics on a dataset
        d = the input data
        posthoc = True if a pair-wise post-hoc test is wanted"""
        # Get combinations of labels so we can see the combinations in the ttest
        results = dict()
        comblabels = [(x,y) for (x,y) in list(itertools.combinations(self.labels,2))]
        
        # Pair-wise post-hoc tests
        if posthoc:
            # Welch's T-test
            if self.is_parametric(d):
                t = [stats.ttest_ind(x,y,equal_var=False) for (x,y) in
                     list(itertools.combinations(d,2))]
            
            # Mann-Whitney U-test
            else:
                t = [stats.mannwhitneyu(x,y) for (x,y) in
                        list(itertools.combinations(d,2))]
                
            # If a test returns significant value, add the type pairs and its
            # corresponding p-value to a dict
            for i in range(len(t)):
                if t[i].pvalue < 0.05/len(t):
                    results[comblabels[i]] = t[i].pvalue
                    
            return results
                
        # Test on the entire dataset
        else:
            if self.is_parametric(d):
                # One-Way ANOVA
                return stats.f_oneway(*d)
            
            # Kruskal-Wallis
            return stats.kruskal(*d)


# Run the analytics
dat = pd.read_csv('bartle.csv', delimiter=",", header=0)
m = Yee(2, dat) # Run the model with bias having double the weight

# Run descriptive statistics
# Calculate player types
m.run()

# Statistical test
testing = m.test(m.distances(m.cluster()), False)
posthoc = m.test(m.distances(m.cluster()), True)

print(testing)
print(posthoc)