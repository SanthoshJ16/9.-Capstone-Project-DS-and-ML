import pandas as pd
class DsFunctions():
    ## Used to read and import csv data 
    ## returns dt object
    def importDataset():
        dt=pd.read_csv("IT_HelpDesk_Random_50k.csv")
        return dt
        
    ## the below method is used to separate Qualitative and Qunatitative data 
    ## returns qual, quan list
    
    def quanQual(dataset):
        qual=[]
        quan=[]
        for columnName in dataset.columns:
            if(dataset[columnName].dtype=="O"):
                qual.append(columnName)
            else:
                quan.append(columnName)
        return quan, qual
        
    ## the below method is used to find Mean , Median , Mode, Percentile, IQR values
    ## returns descriptive table
    def findUnivariate(dataset,quan):
        descriptive=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","Q4:100%",
                                       "IQR","1.5rule","LowerIQR","HigherIQR","Min","Max","kurtosis","skew"],columns=quan)
        for columnName in quan:
            descriptive.loc["Mean",columnName]=dataset[columnName].mean().item()
            descriptive.loc["Median",columnName]=dataset[columnName].median()
            descriptive.loc["Mode",columnName]=dataset[columnName].mode()[0].item()
            descriptive.loc["Q1:25%",columnName]=dataset.describe()[columnName]["25%"].item()
            descriptive.loc["Q2:50%",columnName]=dataset.describe()[columnName]["50%"].item()
            descriptive.loc["Q3:75%",columnName]=dataset.describe()[columnName]["75%"].item()
            descriptive.loc["Q4:100%",columnName]=dataset.describe()[columnName]["max"].item()
            ## IQR Code changes Starts here
            descriptive.loc["IQR",columnName]=descriptive.loc["Q3:75%",columnName]- descriptive.loc["Q1:25%",columnName]
            descriptive.loc["1.5rule",columnName]=1.5 * descriptive.loc["IQR",columnName]
            descriptive.loc["LowerIQR",columnName]= descriptive.loc["Q1:25%",columnName] - descriptive.loc["1.5rule",columnName]
            descriptive.loc["HigherIQR",columnName]= descriptive.loc["Q3:75%",columnName] + descriptive.loc["1.5rule",columnName]
            descriptive.loc["Min", columnName] = dataset.describe().loc["min", columnName]
            descriptive.loc["Max", columnName] = dataset.describe().loc["max", columnName]

            ## Skew , Kurtosis change starts here , Added both columnName in table
            descriptive.loc["kurtosis", columnName] = dataset[columnName].kurtosis()
            descriptive.loc["skew", columnName] = dataset[columnName].skew()

        return descriptive
    
    ## the below method is used to find outliers on the given data set
    ## returns higher outlier columnName list 
    ## returns lower outlier columnName list 
    def checkforOutliers(descriptive,quan):
        lower=[]
        higher=[]
        for columnName in quan:
            if(descriptive.loc["Min",columnName]<descriptive.loc["LowerIQR",columnName]):
                lower.append(columnName)
            if(descriptive.loc["Max",columnName]>descriptive.loc["HigherIQR",columnName]):
                higher.append(columnName)
        return lower, higher
        
    ## the below method is used to replace lower outliers with lower IQR value in dataset and
    ## is used to replace higher outliers with higher IQR value in dataset
    ## returns dataset
    def replaceOutliersWithIQRValues(dataset,descriptive,lower,higher):
        for columnName in lower:
            dataset.loc[dataset[columnName] < descriptive[columnName]["LowerIQR"],columnName] = descriptive[columnName]["LowerIQR"]
        
        ##Below logic Find values above HigherIQR and replaces them with HigherIQR
        for columnName in higher:
            dataset.loc[dataset[columnName] > descriptive[columnName]["HigherIQR"],columnName] = descriptive[columnName]["HigherIQR"]
        return dataset

    #### Code for Standard Normal Distribution Function
    def stdNBgraph(dataset):
        # Coverted to standard Normal Distribution
        import seaborn as sns
        mean=dataset.mean()
        std=dataset.std()
    
        values=[i for i in dataset]
    
        z_score=[((j-mean)/std) for j in values]
    
        sns.histplot(z_score,kde=True)
    
        sum(z_score)/len(z_score)
        #z_score.std()
        return z_score

        
    ## Code for Probability Density Function
    def get_pdf_probability(dataset,startrange,endrange):
        from matplotlib import pyplot
        from scipy.stats import norm
        import seaborn as sns
        import numpy as np
      
    
        ax = sns.distplot(dataset,kde=True,kde_kws={'color':'blue'},color='Green')
        
    
        pyplot.axvline(startrange,color='Red')
        pyplot.axvline(endrange,color='Red')
        # generate a sample
        sample = dataset
        # calculate parameters
        sample_mean =sample.mean()
        sample_std = sample.std()
        print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
        # define the distribution
        dist = norm(sample_mean, sample_std)
    
        # Shade area between start and end - For my own purpose tried this code
        ## x_fill = np.linspace(startrange, endrange, 500)
        ## y_fill = norm.pdf(x_fill, sample_mean, sample_std)
        ## pyplot.fill_between(x_fill, y_fill, color="orange",alpha=0.4,label="Probability Area")
        
        # sample probabilities for a range of outcomes
        values = [value for value in range(startrange, endrange)]
        probabilities = [dist.pdf(value) for value in values]    
        prob=sum(probabilities)
        print("The area between range({},{}):{}".format(startrange,endrange,sum(probabilities)))
        return prob

    ## Creating a Class for frequency Table for the input Column
    ## returns Frequency Table with data unique values, frequency, Relative Frequency and cumulative relative frequency
    def freqTable(dataset,columnName):
        freqTable=pd.DataFrame(columns=["Unique_Values","Frequency","RelativeFrequency","CumSum"])
        freqTable["Unique_Values"]=dataset[columnName].value_counts().index
        freqTable["Frequency"]=dataset[columnName].value_counts().values
        freqTable["RelativeFrequency"]=freqTable["Frequency"]/103 # 103 is the total occurance
        freqTable["CumSum"]=freqTable["RelativeFrequency"].cumsum()
        return freqTable