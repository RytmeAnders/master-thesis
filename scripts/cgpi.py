import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#import yee_classed as yee #For inferential
import itertools

# TODO: Cronbach (= 0.7798)
# TODO: PCA on the questionnaire items
# TODO: Correlate dimension scores with demographic variables

# Yee:
    # Først PCA for at gruppere spørgsmål
    # demographic correlation (regression analysis r^2) på PCA-grupperingerne

class cgpi:
    def __init__(self, d):
        self.d = d
    
    ########## Descriptive
    def describe(self):
        """Calculate all descriptive statistics"""
        age_count = self.d['age'].value_counts(sort=True)
        gender_count = self.d.gender.value_counts(sort=False)
        experience_count = self.d[['playFrequency', 'playAmount']].mean(axis=1).value_counts(sort=False)
        game_count = self.d.game.value_counts(sort=False)
        
        fig, axs = plt.subplots(2,2, figsize=(20,12))       
        # age
        axs[0,0].bar(age_count.index,age_count)
        axs[0,0].set_ylim(0,5)
        axs[0,0].set_title('Age',fontsize=16)
        axs[0,0].set_ylabel('Frequency',fontsize=16)
        
        #Gender
        axs[0,1].bar(gender_count.index,gender_count)
        axs[0,1].set_ylim(0,30)
        axs[0,1].set_title('Gender',fontsize=16)
        axs[0,1].set_ylabel('Frequency',fontsize=16)
        
        #Experience
        axs[1,0].bar(experience_count.index,experience_count)
        axs[1,0].set_ylim(0,10)
        axs[1,0].set_title('Experience',fontsize=16)
        axs[1,0].set_ylabel('Frequency',fontsize=16)
        
        #Player type
        game_labels = ['Single Player', 'Multi Player']
        axs[1,1].bar(game_labels,game_count)
        axs[1,1].set_ylim(0,25)
        axs[1,1].set_title('Type of Game',fontsize=16)
        axs[1,1].set_ylabel('Frequency',fontsize=16)
            
        fig.suptitle("Descriptive statistics of questionnaire test (N={})"
                     .format(len(self.d)), fontsize=30)
        
        plt.show()
    
    ########## CGPI Scores
    def flipScales(self, pt):
        """Flip the negatively worded questionnaire items"""        
        # Convert int to int array
        t = [int(d) for d in str(pt)]
        
        # We need to flip [0,1,2,5,6,7,9,12,13,15]
        #flip_indices = [0,1,2,5,6,7,9,12,13,15]
        flip_indices = [13, 16] # New flip indices after PCA
        
        # Flip indices
        for i in flip_indices:
            t[i] = 8-t[i]
        
        return t
    
    def getDimensions(self, t):
        """Calculate the score of each dimension per player"""
        steps = [0,5,9,13] #defining at which index each dimension begins 
        strides = [5,4,4,4] #first dimension has 5 values, the others have 4
        dims = []
        
        for step, stride in zip(steps,strides):
            dims.append(st.mean(t[step:step+stride]))
        
        return dims
    
    def getFactors(self, t):
        """Calculate mean of factors obtained from PCA"""
        f = []
        facs = []
        # Defining which indices the factors are composed of
        fsteps = [[0,2,4,15],
                  [13,14],
                  [5,6,7,16],
                  [1,10,12],
                  [3,9,11],
                  [8]]
        
        steps = [0,4,6,10,13,16] #defining at which index each dimension begins 
        strides = [4,2,4,3,3,1] #first dimension has 5 values, the others have 4

        # Rearrange the scores so factors are clustered together
        for step in fsteps:
            for i in range(len(step)):
                f.append(t[step[i]])
        
        # Same algorithm as getDimensions, take the mean score for each factor
        for step, stride in zip(steps,strides):
            facs.append(st.mean(f[step:step+stride]))
        
        return facs
    
    def assignToPlayers(self):
        """Assigns the calculated decision dimensions to every participant
        in the dataset. It takes the dimensions (getDimensions), which takes
        the flipped scales (flipScales)"""
        
        self.d['properScales'] = [self.flipScales(x) for x in self.d.type]
        self.d['dimensions'] = [self.getDimensions(self.flipScales(x)) for x in self.d.type]
        self.d['factors'] = [self.getFactors(self.flipScales(x)) for x in self.d.type]
    
        
    def sortPlayers(self):
        """Sort each player into a decision dimension according to argmax of
        their dimension scores"""
        x = []
        
        self.d['highest'] = [np.argmax(x) for x in self.d.factors]
        
        for i in range(6):
            x.append(self.d.loc[self.d.highest == i].reset_index())
                
        return x
    
    ########## Twine
    def countPassages(self):
        """""Count how many times passages have been visited"""""
         # List of all passage names
        names = ["ch1_" + str(i) + "c" for i in range(1,77)]
        names[-2] = "ch1_75" #75 is the last passage, so its has no c after it
        
        # Initialise dictionary of passages and count
        dimcount = dict()
        for name in names:
            dimcount[name] = 0
        
        # Find and count each passage for each participant
        for i in range(len(self.d)):
            for name in names:
                #if path has the passage name, increment that passage
                if self.d.path[i].find(name) != -1:
                    dimcount[name] += 1
        
        # If no one has visited a passage, set the 0 to 1 (so we can divide)
        for name in names:
            if dimcount[name] == 0:
                dimcount[name] = 1
        
        return dimcount
    
    def assignToTwine(self):
        """Give each Twine passage dimension scores"""        
        # List of all passage names
        names = ["ch1_" + str(i) + "c" for i in range(1,77)]
        names[-2] = "ch1_75" #75 is the last passage, so its has no c after it
        
        # Initialise dictionary of passages and dimension scores
        dimscores = dict()
        for name in names:
            dimscores[name] = [0,0,0,0,0,0]
        
         # Assign dimensions scores per participant to each passage
        for i in range(len(self.d)):
            for name in names:
                #if path has the passage name, add the participants' dimension
                #scores to that passage
                if self.d.path[i].find(name) != -1:
                    dimscores[name] = np.add(dimscores[name], self.d.factors[i])
        
        # Get the average dimension scores
        # (dividing by the number of people who visited that passage)
        counts = self.countPassages()
        for name in names:
            dimscores[name] = [i/counts[name] for i in dimscores[name]]
        
        return dimscores
    
    def getPassage(self, name):
        """Gets all participants who visited a certain passage"""
        out = []
        
        for i in range(len(self.d)):
            if self.d.path[i].find(name) != -1:
                out.append(self.d.factors[i])
        
        return out

    ########## Continuation Desire
    def getAverageCD(self, cds):
        """Calculate the average continuation desire per pivotal story
        point"""
        
        # Array of all continuation desires into arrays
        t = []
        for cd in cds:
            t.append([int(d) for d in str(cd)])
        
        # We sum along the first axis (t,0), and take the average of each index
        y = [i/len(cds) for i in np.sum(t,0)]
        
        return y
    
    def plotCD(self, pdata):
        """Get a graph of average continuation desire for the story"""
        x = ['Before start', 'First item', 'Meeting the group', 'The End']
        
        plt.plot(x, pdata)
        plt.ylim(1,7)
        plt.title('Average Continuation Desire In Narrative')
        plt.ylabel('Likert scale')
    
    def inferCD(self, cds):
        """Do inferential statistics on continuation desire. We try to find
        a significant difference between discrete points in the narrative"""
        t = []
        
        # Convert int to int array
        for cd in cds:
            t.append([int(d) for d in str(cd)])
        
        # Remove lists without all four indices
        t = [i for i in t if len(i) == 4]
        
        # Group cds that share the time in the narrative (group all the first
        # indices, then the second, and so on)
        cd_groups = []
        for index in range(4):
            cd_groups.append([i[index] for i in t])
        
        return cd_groups
    
    
    def extract(self, i, lst): 
        """"Gets the ith index of sublists in lst"""
        return [item[i] for item in lst]
    
    def printDict(self):
        """Prints the decision dimension dictionaries"""
        dims = m.assignToTwine()
        
        for name in dims:
            for y in dims[name]:
                print(str(name) + ": " + str(dims[name][y]))

plt.style.use('seaborn') # pretty matplotlib plots
dat = pd.read_csv('cgpi.csv', delimiter=',', header=0)

m = cgpi(dat)
m.assignToPlayers() # Calculate each players dimension scores
dimScores = m.assignToTwine() # Assign dimension scores to each passage in Twine
dimCounts = m.countPassages() # Count how many times each passage has been visited

# Continuation desire
m.plotCD(m.getAverageCD(dat.cd[7:])) #Plot for participants with all indices

sortedPlayers = m.sortPlayers()

m.describe()

############ Check if single player / multi player has an effect on the sample
single = dat.loc[dat.game == 0]
multi = dat.loc[dat.game == 1]

del single.age[26] #26 is nan

# Check for parametricity (it is not)
normal_sage = stats.shapiro(single.age)
normal_mage = stats.shapiro(multi.age)

#significant (U=40.5, p=0.004)
mannu = stats.mannwhitneyu(single.age, multi.age)

# Let's plot the correlation between age and mode of play
#g = single.age

list1 = [x for x in single.age[::2]]
list2 = [x for x in single.age[1::2]]
g = [(x+y)/2 for x,y in zip(list1,list2)]
g = np.array(g)
h = multi.age

#h = np.repeat(multi.age, 2)
#slope = -0.099

#h = dat.age
#g = dat.game

#plt.plot(g, h, 'o')
#m, b = np.polyfit(g, h, 1)
#plt.plot(g, m*g + b)
#plt.xlabel('Single player age')
#plt.ylabel('Multi player age')
#plt.title("Correlation age/preferred mode of play (r=-0.09)")

#corr_age = np.corrcoef(g,h)

# Mode of play versus gender
#ctable = [[11,10],
#          [0,10]]
#chi = stats.chi2_contingency(ctable)


# Mode of play versus experience (PARAMETRIC!)
#exp_single = single[['playFrequency', 'playAmount']].mean(axis=1)
#exp_multi = multi[['playFrequency', 'playAmount']].mean(axis=1)

#ttest = stats.ttest_ind(exp_single,exp_multi)


# For dims_new cronbach = 0.7614 = good reliability
#x = []
#for scale in dat.properScales:
#    x.append(scale)

#h = pd.DataFrame(x) # Turn the list of types into a dataframe
#h.to_csv("./dims_new.csv", sep=',',index=False) # Write that dataframe to a csv
