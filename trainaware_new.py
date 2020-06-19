import pandas as pd
import numpy as np

import streamlit as st
import xgboost as xgb

#once you import data into streamlit, DO NOT DO ANYTHING THAT WILL MUTATE IT!!!!!
#setting random number seed
myfavoritenumber = 13
seed = myfavoritenumber
np.random.seed(seed)


X_train = st.cache(pd.read_csv)("train_features.csv") #the .. goes one directory up (two periods)
Y_train = st.cache(pd.read_csv)("train_labels.csv") #the .. goes one directory up (two periods)
X_test = st.cache(pd.read_csv)("test_features.csv")
Y_test = st.cache(pd.read_csv)("test_labels.csv")


st.title("TrainAware: Work Smarter, Not Harder")

st.write("This application is designed to help you identify when you might be overtraining, so that you can reduce your risk of work-out related injury")

st.write("Input the information below from your wearable fitness tracker. Defaults are population averages.")

#kcal = st.slider("In the last week, how any calories have you burned as a result of activity? (kcal)", (int((X_test.active_calories_burned_norm.min() * 624.7314961807097)+500.11064030131826)),(int((X_test.active_calories_burned_norm.max() * 624.7314961807097)+500.11064030131826)))
kcal = st.slider("In the last week, how any calories have you burned as a result of activity? (kcal)", (int((X_test.active_calories_burned_norm.min() * 624.7314961807097)+500.11064030131826)),(int((X_test.active_calories_burned_norm.max() * 624.7314961807097)+500.11064030131826)),500)

#ACWR = st.slider("What was your acute-to-chronic workload ratio for this week? (calculate from number of calories burned or input from fitness tracker)",(int((X_test.ACWR_norm.min()*0.5789888501236283)+1.0449098150720992)),int((X_test.ACWR_norm.max()*0.5789888501236283)+1.0449098150720992))

ACWR = st.slider("What was your acute-to-chronic workload ratio for this week? (calculate from number of calories burned or input from fitness tracker)",((X_test.ACWR_norm.min()*0.5789888501236283)+1.0449098150720992),((X_test.ACWR_norm.max()*0.5789888501236283)+1.0449098150720992), 1.044)
body_temp = st.slider("What was your average body temperature yesterday (Fahrenheit)?", (int(((X_test.body_temperature_avg_norm.min()*0.2149352716004961)+36.360099254590985)*1.8)+32),(int(((X_test.body_temperature_avg_norm.max()*0.2149352716004961)+36.360099254590985)*1.8)+32), 97)
bpm =st.slider("What was your average heart rate yesterday? (bpm)", (int((X_test.bpm_norm.min() * 10.439030945435983)+77.36488825936418)),(int((X_test.bpm_norm.max()*10.439030945435983)+77.36488825936418)),77)
height = st.slider("What is your height (in inches)?", (int((X_test.height_in_norm.min()*2.5519672115618115)+67.1371419981172)), (int((X_test.height_in_norm.max()*2.5519672115618115)+67.1371419981172)),67)
weight = st.slider("What is your weight (in pounds)?", (int((X_test.weight_lbs_norm.min()*41.819657211176384)+175.28163438588976)), (int((X_test.weight_lbs_norm.max()*41.819657211176384)+175.28163438588976)),175)
bmi = st.slider("What is your body mass index (bmi)?", (int((X_test.BMI_norm.min()*6.764111088344532)+27.353030063051644)), (int((X_test.BMI_norm.max()*6.764111088344532)+27.353030063051644)),27)
pulse_max  = st.slider("In the past week, what was your maximum pulse?", (int((X_test.pulse_max_norm.min()*26.630131193251586)+104.35805084745765)), (int((X_test.pulse_max_norm.max()*26.630131193251586)+104.35805084745765)),104)

data = {'gender_f': 1,
       'gender_m': 0,
       'age_range_18-24': 0,
       'age_range_25-34': 0,
       'age_range_35-44': 0,
       'age_range_45-54': 1,
       'age_range_55-64': 0,
       'age_range_65-74': 0,
       'active_calories_burned_norm': (kcal - 500.11064030131826)/624.7314961807097,
       'steps_count_norm': -0.2461440515610692,
       'steps_speed_norm': -0.13744036247105884,
       'body_temperature_avg_norm': (body_temp - 36.360099254590985)/0.2149352716004961,
       'pulse_average_norm': 0.025230906947202696,
       'stand_hours_total_norm': 0.0508945259098841,
       'total_number_of_flights_climbed_norm': 0.023871369638159775,
       'pulse_min_norm': -0.16767101326046926,
       'pulse_max_norm': (pulse_max - 104.35805084745765)/26.630131193251586,
       'average_spo2_value_norm': -0.7006572853868199,
       'distance_mi_norm': 0.15212434682828463,
       'ACWR_norm': (ACWR - 1.0449098150720992)/0.5789888501236283,
       'height_in_norm': (height - 67.1371419981172)/2.5519672115618115,
       'weight_lbs_norm': (weight - 175.28163438588976)/41.819657211176384,
       'bpm_norm': bpm,
       'heart_rate_norm': 0.01439997496980094,
       'BMI_norm': -0.0882595844071344,
       'distance_yds_norm': 0.1521243468282846}

dict = pd.DataFrame.from_dict(data, orient="index")


cols = dict.transpose()

# Create the model with 100 trees
D_train = xgb.DMatrix(X_train, label= Y_train)
#D_test = xgb.DMatrix(cols, label=Y_test)
D_test = xgb.DMatrix(cols)
#best parameters from CV:
param = {
    'eta': 0.2,
    'gamma': 0.1,
    'max_depth': 4,
    'min_child_weight': 5,
    'objective': 'multi:softprob',
    'num_class': 2}

steps = 20  # The number of training iterations

model = xgb.train(param, D_train, steps)

predictions = model.predict(D_test)[0]

percent = list(predictions)[1] * 100
rounded_per = round(percent)

if st.button("What's my TrainAware score?"):
    st.header("Your chance of overtraining next week is {}%".format(rounded_per)) #predictions object
    if rounded_per > 60:
        st.header("Your chance of overtraining is high! Consider taking some rest days this week to reduce your chance of overtraining.")
        st.header("To increase your flexibility and further reduce your chance of injury due to overtraining, consider stretching after your workout. Select a muscle group you want to stretch below")
    elif rounded_per > 30 and rounded_per < 60:
        st.header("Your chance of overtraining is medium! It's safe to continue exercising at your current pace, just don't increase the intensity too much! You might consider stretches to further reduce your chance of overtraining")
    else:
        st.header("Your chance of overtraining is low! Train away!")

if rounded_per > 30:
    selected = st.selectbox("What area of your body have you been working out hard? (Select AFTER you get your TrainAware score)",('<select>','Triceps','Biceps','Forearms','Shoulders','Back', 'Chest','Abs','Glutes','Quads','Hip Flexors', 'Hamstrings','Calves'))
    if selected == 'Triceps':
        st.write("Consider an Overhead Tricep Stretch")
        st.write("Instructions: Stand tall with good posture. Reach your left arm up into the air over your head, then bend your elbow, placing your left hand flat on your upper back, as flexibility allows. \
                With your right hand, grasp your left arm, just above the elbow, and use your right hand to lightly pull your left elbow toward your head as your left hand reaches farther down your back. You should feel a stretch through your left triceps. \
                When you feel a good stretch, hold for 20 to 30 seconds before switching sides.")
    elif selected == 'Biceps':
        st.write("Consider a Standing Bicep Stretch")
        st.write("Stand tall, feet shoulder-distance apart, knees slightly bent. With your arms extended, clasp your hands directly behind your back, palms touching. \
                From this position, rotate your wrists backward, opening your still-clasped hands so your palms are facing the ground. \
                Keeping your elbows straight, raise your arms behind your body until you feel a stretch through your biceps. Hold for 20 to 30 seconds, release, and repeat. To get a deeper stretch through your shoulders and chest, look up toward the ceiling and draw your shoulders backward to broaden your chest.")
    elif selected == 'Forearms':
        st.write("Consider an Alternating Wrist Pull")
        st.write("From a seated or standing position, extend both arms directly in front of your chest. Flex your left wrist toward you, so your fingers point upward, then rotate it to the outside until your fingers point toward the floor. \
                Clasp your left palm with your right hand and lightly pull your left hand toward you until you feel a stretch through your left forearm. Hold for 10 seconds, then switch sides. Repeat two to three times per side.")
    elif selected == 'Shoulders':
        st.write("Consider a Cross-body shoulder stretch")
        st.write("Stand with your feet shoulder-width apart, knees slightly bent, both arms extended down at your sides. \
                Keeping your left elbow straight, raise your left arm directly in front of your chest, then reach it across your body, toward your right shoulder. \
                Bend your right elbow and clasp your left arm just below the elbow with your right hand. Use your right hand to gently press your left arm closer to your chest. When you feel a good stretch through your shoulder and into your triceps, hold the position for 20 to 30 seconds before switching sides.")
    elif selected == 'Back':
        st.write("Consider a Cat, cow, child's pose sequence")
        st.write("Start on your hands and knees on an exercise mat, your palms positioned under your shoulders, your knees positioned under your hips, and your back and neck in a flat (neutral) position. \
                Enter cow pose as you inhale by looking up and stretching your chest forward as you simultaneously press your hips upward, trying to articulate your tailbone toward the ceiling. This action will cause your low back to dip toward the floor, stretching your abs in the process. Hold for a count of three. \
                Enter cat pose as you exhale by lowering your head between your shoulders and tucking your tailbone under, scooping your hips forward and pressing your upper back and shoulders toward the ceiling. Round your shoulders outward to broaden the stretch across your upper back. Use your abs to keep your hips scooped forward to feel a nice stretch through your low back and hips. Hold for a count of three. \
                Return to a neutral, tabletop-like position on your next inhale, then as you exhale, enter child's pose by sitting back on your heels, your chest folding over your quads as you reach your arms as far as you can over your head, your palms flat on the ground. Hold this position for three deep breaths, really sinking your hips into your heels with each exhale. \
                After three breaths, return to the tabletop position and perform three to five more rounds of the sequence.")
    elif selected == 'Chest':
        st.write("Consider a Wall-assisted single-arm chest stretch")
        st.write("Stand with your left side perpendicular to a wall, about 1 to 2ft away from the wall. \
                Reach your left arm behind you, your elbow extended, and place your left palm flat on the wall. Start with your hand positioned so your left arm is parallel to the floor. \
                You should feel a stretch through your chest and the front of your shoulder. To deepen the stretch, shift your weight toward the wall, or slowly move your feet closer to the wall. When you feel a good stretch, hold the position for 20 to 30 seconds before switching sides. \
                You can change the angle of the stretch by positioning your palm higher or lower on the wall.")
    elif selected == 'Abs':
        st.write("Consider a Stability ball-assisted backbend")
        st.write("Sit on a stability ball, your feet planted shoulder-distance apart, your knees bent at 90-degree angles. \
                Slowly step forward as you simultaneously lean back, lying down on top of the ball. \
                With your knees still bent, extend your arms over your head and reach them backward, allowing your back to arch with the support of the ball. You should feel a good stretch through your abs, chest, and shoulders. \
                Continue reaching backward as far as you can, and if you feel comfortable, step your feet forward, extending your knees to enjoy a stretch through your quads and hip flexors. \
                Hold the supported backbend for 30 to 60 seconds, breathing deeply throughout.")
    elif selected == 'Glutes':
        st.write("Consider a Supine figure-4 glute stretch")
        st.write("Lie on your back, your knees bent, feet flat on the floor. Cross your right ankle over your left knee, as if creating a '4' with your legs. \
                Contract your abs to keep your low back in contact with the ground, then lift your left foot from the floor. Reach both arms forward to grasp the back of your left thigh. \
                From this position, use your arms to pull your left leg closer to your body to deepen the stretch. Keep your right knee pointed toward the side of the room -- don't let it adduct toward your midline. If you're flexible enough, you can use your right elbow to press your right knee away. When you feel a deep stretch through your right glute, hold the position for 20 to 30 seconds before switching sides.")
    elif selected == 'Quads':
        st.write("Consider a Standing quad stretch")
        st.write("Stand tall, feet planted roughly hip-distance apart, knees slightly bent. \
                Shift your weight to your left foot, bend your right knee behind you, and lift your right foot toward your same-side glute. \
                Grasp the top of your right foot with your right hand to pull your right heel closer to your glute. Make sure your right knee is pointing toward the ground, your right leg still aligned with your left leg. When you feel a good stretch, hold the position for 20 to 30 seconds before switching legs.")
    elif selected == 'Hip Flexors':
        st.write("Consider a Standing hip flexor stretch")
        st.write("Stand tall, your feet roughly hip-distance apart, your knees slightly bent. \
                Take a wide step forward with your right foot, both feet solidly planted on the ground. \
                Keeping your chest and torso tall, bend both knees slightly and sink your hips down an inch or two, as if entering a 'baby lunge'. \
                From this position, tuck your hips under and press them forward until you feel a stretch through the front of your left hip. Hold for 20 to 30 seconds before switching sides.")
    elif selected == 'Hamstrings':
        st.write("Consider a Strap-assisted supine hamstring stretch")
        st.write("Sit on a mat, your knees bent, feet flat on the floor. Position the middle of your strap under your right foot, then hold the ends of the strap in each hand. \
                Lie back on the mat, then lift your right foot, pressing it against the strap. Extend your right leg fully. \
                Draw your right leg up and toward your body, keeping your knee straight. Use your hands to pull the strap lightly and deepen the stretch. When you feel a good stretch, hold the position for 20 to 30 seconds before switching sides.")
    elif selected == 'Calves':
        st.write("Consider a Wall-assisted calf stretch")
        st.write("Stand facing a wall, your feet hip-distance apart. Press your palms into the wall at chest height, then lunge your right foot backward, planting your foot on the ground, bending your front knee as needed, but keeping your back knee straight. \
                With your torso straight, lean forward into the wall, stretching the back of your right calf. When you feel a good stretch, hold for 20 to 30 seconds before switching sides.")

if st.checkbox('view the simple data'):
    st.write(X_train)
