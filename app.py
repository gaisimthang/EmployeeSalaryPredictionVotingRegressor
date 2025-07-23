from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("best_salary_model.pkl")


gender_map = {"Male": 1, "Female": 0}
degree_map = {"High School": 4, "Bachelor": 1, "Master": 2, "PhD": 3}


job_freq = {title: i+1 for i, title in enumerate([
    'Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Director', 'Marketing Analyst',
    'Product Manager', 'Sales Manager', 'Marketing Coordinator', 'Senior Scientist', 'Software Developer',
    'HR Manager', 'Financial Analyst', 'Project Manager', 'Customer Service Rep', 'Operations Manager',
    'Marketing Manager', 'Senior Engineer', 'Data Entry Clerk', 'Sales Director', 'Business Analyst',
    'VP of Operations', 'IT Support', 'Recruiter', 'Financial Manager', 'Social Media Specialist', 'Software Manager',
    'Junior Developer', 'Senior Consultant', 'Product Designer', 'CEO', 'Accountant', 'Data Scientist',
    'Marketing Specialist', 'Technical Writer', 'HR Generalist', 'Project Engineer', 'Customer Success Rep',
    'Sales Executive', 'UX Designer', 'Operations Director', 'Network Engineer', 'Administrative Assistant',
    'Strategy Consultant', 'Copywriter', 'Account Manager', 'Director of Marketing', 'Help Desk Analyst',
    'Customer Service Manager', 'Business Intelligence Analyst', 'Event Coordinator', 'VP of Finance',
    'Graphic Designer', 'UX Researcher', 'Social Media Manager', 'Director of Operations', 'Senior Data Scientist',
    'Junior Accountant', 'Digital Marketing Manager', 'IT Manager', 'Customer Service Representative',
    'Business Development Manager', 'Senior Financial Analyst', 'Web Developer', 'Research Director',
    'Technical Support Specialist', 'Creative Director', 'Senior Software Engineer', 'Human Resources Director',
    'Content Marketing Manager', 'Technical Recruiter', 'Sales Representative', 'Chief Technology Officer',
    'Junior Designer', 'Financial Advisor', 'Junior Account Manager', 'Senior Project Manager', 'Principal Scientist',
    'Supply Chain Manager', 'Senior Marketing Manager', 'Training Specialist', 'Research Scientist',
    'Junior Software Developer', 'Public Relations Manager', 'Operations Analyst', 'Product Marketing Manager',
    'Senior HR Manager', 'Junior Web Developer', 'Senior Project Coordinator', 'Chief Data Officer',
    'Digital Content Producer', 'IT Support Specialist', 'Senior Marketing Analyst', 'Customer Success Manager',
    'Senior Graphic Designer', 'Software Project Manager', 'Supply Chain Analyst', 'Senior Business Analyst',
    'Junior Marketing Analyst', 'Office Manager', 'Principal Engineer', 'Junior HR Generalist',
    'Senior Product Manager', 'Junior Operations Analyst', 'Senior HR Generalist', 'Sales Operations Manager',
    'Senior Software Developer', 'Junior Web Designer', 'Senior Training Specialist', 'Senior Research Scientist',
    'Junior Sales Representative', 'Junior Marketing Manager', 'Junior Data Analyst',
    'Senior Product Marketing Manager', 'Junior Business Analyst', 'Senior Sales Manager',
    'Junior Marketing Specialist', 'Junior Project Manager', 'Senior Accountant', 'Director of Sales',
    'Junior Recruiter', 'Senior Business Development Manager', 'Senior Product Designer',
    'Junior Customer Support Specialist', 'Senior IT Support Specialist', 'Junior Financial Analyst',
    'Senior Operations Manager', 'Director of Human Resources', 'Junior Software Engineer',
    'Senior Sales Representative', 'Director of Product Management', 'Junior Copywriter',
    'Senior Marketing Coordinator', 'Senior Human Resources Manager', 'Junior Business Development Associate',
    'Senior Account Manager', 'Senior Researcher', 'Junior HR Coordinator', 'Director of Finance',
    'Junior Marketing Coordinator', 'Junior Data Scientist', 'Senior Operations Analyst',
    'Senior Human Resources Coordinator', 'Senior UX Designer', 'Junior Product Manager',
    'Senior Marketing Specialist', 'Senior IT Project Manager', 'Senior Quality Assurance Analyst',
    'Director of Sales and Marketing', 'Senior Account Executive', 'Director of Business Development',
    'Junior Social Media Manager', 'Senior Human Resources Specialist', 'Senior Data Analyst',
    'Director of Human Capital', 'Junior Advertising Coordinator', 'Junior UX Designer', 'Senior Marketing Director',
    'Senior IT Consultant', 'Senior Financial Advisor', 'Junior Business Operations Analyst',
    'Junior Social Media Specialist', 'Senior Product Development Manager', 'Junior Operations Manager',
    'Senior Software Architect', 'Junior Research Scientist', 'Senior Financial Manager', 'Senior HR Specialist',
    'Senior Data Engineer', 'Junior Operations Coordinator', 'Director of HR', 'Senior Operations Coordinator',
    'Junior Financial Advisor', 'Director of Engineering', 'Software Engineer Manager', 'Back end Developer',
    'Senior Project Engineer', 'Full Stack Engineer', 'Front end Developer', 'Front End Developer',
    'Director of Data Science', 'Human Resources Coordinator', 'Junior Sales Associate', 'Human Resources Manager',
    'Juniour HR Generalist', 'Juniour HR Coordinator', 'Digital Marketing Specialist', 'Receptionist',
    'Marketing Director', 'Social Media Man', 'Delivery Driver'
])}

job_titles = list(job_freq.keys())

@app.route('/')
def home():
    return render_template('index.html', job_titles=job_titles)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        experience = float(request.form['experience'])
        gender = request.form['gender']
        degree = request.form['degree']
        job_title = request.form['job_title']

        encoded_input = {
            'Age': age,
            'Experience_Years': experience,
            'Gender_Encode': gender_map.get(gender, 0),
            'Degree_Encode': degree_map.get(degree, 0),
            'Job_Title_Encode': job_freq.get(job_title, 0)
        }

        X = pd.DataFrame([encoded_input])
        predicted_salary = model.predict(X)[0]

        return render_template('index.html', prediction_text=f"Predicted Salary: ₹{predicted_salary:,.2f}", job_titles=job_titles)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}", job_titles=job_titles)

if __name__ == '__main__':
    app.run(debug=True)
