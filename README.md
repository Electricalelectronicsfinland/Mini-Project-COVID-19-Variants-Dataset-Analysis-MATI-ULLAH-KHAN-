# Mini-Project-COVID-19-Variants-Dataset-Analysis-MATI-ULLAH-KHAN-#
# Mini-Project :  COVID-19 Variants Dataset Analysis-METROPOLIA UNIVERSITY OF APPLIED SCIENCES#
# ALL RIGHTS ARE RESERVED ABOUT THE PROJECT REPORT, CODING AND AND POWER POINT SLIDES AND REFERENCE MATERIALS#
# READERS CAN READ, AND UNDERSATND THE ATTCHED FILES FOR CITATIMNG THEIR RESEARCH WORK BUT CAN BE DIRECTLY COPIED FROM MY GITHUB ITS STRICKLY FORBIDDEN#
#####################################
Report: Understanding of Each Task
###########################################
1. General Task Explanations (Step-by-Step)
###########################################
Task No.	What the Task Is	How It’s Solved
1	Load and Explore Data	Read surv_variants.csv, check columns, shape.
2	Clean Data	Fix missing values, prepare dates, calculate durations.
3	Exploratory Data Analysis (EDA)	Use histograms, boxplots, bar plots, scatter plots.
4	Statistical Testing	Perform two-sample t-test for mortality rates.
5	Correlation Analysis	Calculate Pearson correlation and plot heatmap.
6	Survival and Variant Spread Analysis	Calculate durations, censored flags, and timelines.
7	Growth Rate Analysis	Analyse growth rates over time per variant.
8	Predictive Modelling	Regression models to predict mortality rates.

#########################
2. About the Python Code 
#########################
Part	Code Description	How It Solves the Question
Loading Data	pd.read_csv()	Load dataset into DataFrame.
Data Cleaning	pd.to_datetime(), .fillna()	Fix dates and missing values.
Plotting	matplotlib, seaborn	Create histograms, scatter plots, line plots.
Statistical Test	scipy.stats.ttest_ind()	Compare two groups (censored vs uncensored mortality rates).
Correlation	.corr(), sns.heatmap()	Find relationships between variables.
Regression Models	LinearRegression, RandomForestRegressor	Predict mortality rates from features.

###############################
3. Comments on Each Visual Plot 
################################

Plot	Description	Insights
Variant Distribution	Bar plot of variant counts.	Alpha and Delta dominate.
Country Sequences	Bar plot of total sequences per country.	USA, UK lead sequencing efforts.
Histogram of Mortality Rate	Distribution of mortality rates.	Most under 5%.
Histogram of Duration	How long variants last.	100-400 days common.
Correlation Heatmap	Variables inter-relationship.	Deaths highly correlated to cases.
Boxplot: Censored vs Uncensored Mortality	Compare mortality between groups.	No major difference.
Scatter Plot Mortality vs Duration	Mortality over time of detection.	No clear trend — widely spread.
Time Series Decomposition	Seasonal trends and noise in variant counts.	Clear rise/fall patterns.

##################################
4. Data Sets and Parameters Used
##################################
Parameter	Meaning	Usage
country	Country name	Where sequences are reported.
first_seq	First detection date	Start of variant timeline.
last_seq	Last detection date	End of variant timeline.
num_seqs	Number of sequences	Variant popularity or spread.
variant	Variant name	Alpha, Delta, etc.
censure_date	Censoring point	Survival analysis cutoff.
duration	Duration in days	Active time of variant.
censored	1 if censored, 0 otherwise	For survival analysis.
mortality_rate	Deaths / Cases (%)	Key outcome variable.
total_cases	COVID-19 cases	To measure impact.
total_deaths	COVID-19 deaths	For mortality calculation.
growth_rate	Speed of case increase	How fast variant spread.

6. Visual Plots Trends and Insights 

Plot	What It Shows	Trends
Mortality Histogram	Mortality distribution	Most countries had low mortality.
Boxplot Sequence Count	Outliers in sequence counts	Some countries sequenced a lot more.
Total Sequences by Month	Variant spread across months	Peaks seen corresponding to Delta spread.
Monthly Mortality Trends	Change over time	Slight increase during Delta wave.
Top 12 Variants	By geographical spread	Delta globally dominant.
Timeline of Delta Variant	Spread timeline	Rapid rise in mid-2021.
Fastest Spreading Variants	Top 8 variants	Delta, Omicron among fastest.
Avg. Variant Duration by Country	Variants survival times	Similar across regions.
Mortality Over Time	Mortality evolution	Minor peaks seen during variant waves.
Growth Rate by Variant	Growth comparison	Delta had faster growth than others.
Time Series Decomposition	Seasonal trend decomposition	Clear COVID waves.
Variant First Appearance vs Spread	Relationship between detection and spread	Early detection variants had wider spread.
Mortality vs Time Since Detection	Scatter plot	No simple trend.
Global Variant Timeline	Global emergence visualized	Rise and fall visible.
World Map Variant Spread	Global distribution	Delta was everywhere.
Mortality by Variant	Boxplots by variant	Some variants more deadly.
Growth Over Time (Normal and Log)	Line plots	Clear exponential growth visible.
Pie Charts Comparison	Variant share pie charts	Clear dominance shifts over time.
Mortality vs Growth Rate	Scatter plot	Weak relationship.
Top Countries by Sequencing	Bar plot	USA, UK, Germany, India lead.
Variant Interactive Tracer	Plotly interactive plot	Exploratory deep dive into variant dynamics.
Heatmap of Features	Feature interrelations	Mortality tied strongly to deaths, cases.
Top 20 Features for Prediction	Bar plot of feature importance	Deaths, cases dominate prediction.
Regression Models	Predict mortality	Models compared using R² score.

6. Research Questions and Answers

Research Question	Answer
Which variants dominated globally?	Alpha and Delta.
Which countries sequenced the most?	USA, UK, India.
What was the mortality rate?	Mostly under 5%, higher during Delta waves.
Is mortality linked to variant duration?	Not strongly linked.
How fast did variants spread?	Delta and Omicron spread fastest.
Which features predict mortality rate best?	Total cases, deaths, variant type.
Is there seasonal trend in variant emergence?	Yes, waves correspond to seasons.
#################
  Final Summary
#################
This project systematically analysed the global spread, impact, and dynamics of COVID-19 variants using real-world sequencing data.

It involved data cleaning, exploratory analysis, advanced plotting, statistical hypothesis testing, survival analysis, and predictive modelling.
#############
Key Outcomes:
#############
•	Alpha and Delta were the globally dominant variants.
•	Mortality rates remained relatively low overall.
•	Time-based trends showed clear COVID-19 waves matching variant dominance.
•	Growth rates varied by variant, with Delta and Omicron being fastest.
•	Predictive modelling shows mortality strongly tied to case counts and deaths.

###########
Conclusion:
###########
Understanding variant spread and impact remains critical for future pandemic preparedness.

