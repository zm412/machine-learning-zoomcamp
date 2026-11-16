KNOWN_SKILLS = [
    "accounting",
    "agile",
    "algorithms",
    "apis",
    "aws",
    "backup and recovery",
    "budgeting",
    "communication",
    "css",
    "data structures",
    "data visualization",
    "database design",
    "deep learning",
    "docker",
    "excel",
    "financial modeling",
    "git",
    "hr analytics",
    "html",
    "incident response",
    "java",
    "javascript",
    "kubernetes",
    "linux",
    "machine learning",
    "mlops",
    "networking",
    "pandas",
    "performance tuning",
    "postgresql",
    "power bi",
    "process mapping",
    "project planning",
    "python",
    "pytorch",
    "react",
    "requirements analysis",
    "risk analysis",
    "risk management",
    "security auditing",
    "siem",
    "sql",
    "stakeholder management",
    "statistics"
]

def normalize_record(rows):
    dicts = []
    for row in rows:
        job = str(row['job_title']).strip().lower().replace(' ', '_')
        
        row_dict = {
            'job_title': row['job_title'],
            'experience_years': row['experience_years'],
            'education': row['education']
        }

        candidate_skills = {s.strip().lower() for s in str(row['skills']).split(',') if s.strip()}

        for skill in KNOWN_SKILLS:
            if skill in candidate_skills:
                feature_name = f"job__{job}__skill__{skill.replace(' ', '_')}"
                row_dict[feature_name] = 1

        dicts.append(row_dict)
    return dicts
