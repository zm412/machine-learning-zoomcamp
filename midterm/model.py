KNOWN_SKILLS = [
    'stakeholder management',
    'aws',
    'excel',
    'accounting',
    'css',
    'python',
    'kubernetes',
    'documentation',
    'project planning',
    'incident response',
    'git',
    'algorithms',
    'hr analytics',
    'research',
    'communication',
    'html',
    'power bi',
    'agile',
    'performance tuning',
    'machine learning',
    'react',
    'teamwork',
    'networking',
    'problem solving',
    'javascript',
    'postgresql',
    'siem',
    'database design',
    'apis',
    'mlops',
    'linux',
    'statistics',
    'sql',
    'deep learning',
    'docker',
    'pytorch',
    'financial modeling',
    'testing',
    'data visualization',
    'budgeting',
    'presentation',
    'java',
    'security auditing',
    'time management',
    'risk analysis',
    'data structures',
    'process mapping',
    'risk management',
    'backup and recovery',
    'requirements analysis',
    'leadership',
    'pandas'
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
