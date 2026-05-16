-- =====================================================
-- JOB ACCEPTANCE PREDICTION SYSTEM
-- SQL ANALYTICAL QUERIES
-- =====================================================

-- =====================================================
-- CREATE DATABASE
-- =====================================================

CREATE DATABASE IF NOT EXISTS job_acceptance_db;

-- =====================================================
-- USE DATABASE
-- =====================================================

USE job_acceptance_db;

-- =====================================================
-- VIEW TABLE STRUCTURE
-- =====================================================

DESCRIBE final_data;

-- =====================================================
-- VIEW SAMPLE RECORDS
-- =====================================================

SELECT *
FROM final_data
LIMIT 10;

-- =====================================================
-- TOTAL CANDIDATE COUNT
-- =====================================================

SELECT
    COUNT(*) AS total_candidates
FROM final_data;

-- =====================================================
-- GENDER DISTRIBUTION
-- =====================================================

SELECT
    gender,
    COUNT(*) AS total_candidates
FROM final_data
GROUP BY gender;

-- =====================================================
-- PLACEMENT DISTRIBUTION
-- =====================================================

SELECT
    status,
    COUNT(*) AS total_candidates
FROM final_data
GROUP BY status;

-- =====================================================
-- AVERAGE TECHNICAL SCORE
-- =====================================================

SELECT
    AVG(technical_score) AS avg_technical_score
FROM final_data;

-- =====================================================
-- AVERAGE SKILLS MATCH PERCENTAGE
-- =====================================================

SELECT
    AVG(skills_match_percentage) AS avg_skills_match
FROM final_data;

-- =====================================================
-- DEGREE SPECIALIZATION ANALYSIS
-- =====================================================

SELECT
    degree_specialization,
    COUNT(*) AS total_candidates
FROM final_data
GROUP BY degree_specialization
ORDER BY total_candidates DESC;

-- =====================================================
-- PLACEMENT RATE BY GENDER
-- =====================================================

SELECT
    gender,
    AVG(status) * 100 AS placement_rate
FROM final_data
GROUP BY gender;

-- =====================================================
-- TOP 10 HIGH SKILL CANDIDATES
-- =====================================================

SELECT
    gender,
    technical_score,
    skills_match_percentage
FROM final_data
ORDER BY skills_match_percentage DESC
LIMIT 10;

-- =====================================================
-- EXPERIENCE CATEGORY ANALYSIS
-- =====================================================

SELECT

    CASE

        WHEN years_of_experience < 2
        THEN 'Fresher'

        WHEN years_of_experience < 5
        THEN 'Junior'

        ELSE 'Senior'

    END AS experience_category,

    COUNT(*) AS total_candidates

FROM final_data

GROUP BY experience_category;

-- =====================================================
-- INTERVIEW PERFORMANCE ANALYSIS
-- =====================================================

SELECT
    interview_level,
    COUNT(*) AS total_candidates
FROM final_data
GROUP BY interview_level;

-- =====================================================
-- SKILLS LEVEL ANALYSIS
-- =====================================================

SELECT
    skills_level,
    COUNT(*) AS total_candidates
FROM final_data
GROUP BY skills_level;

-- =====================================================
-- COMPANY TIER ANALYSIS
-- =====================================================

SELECT
    company_tier,
    COUNT(*) AS total_candidates
FROM final_data
GROUP BY company_tier;

-- =====================================================
-- JOB ROLE MATCH ANALYSIS
-- =====================================================

SELECT
    job_role_match,
    COUNT(*) AS total_candidates
FROM final_data
GROUP BY job_role_match;

-- =====================================================
-- RELOCATION WILLINGNESS ANALYSIS
-- =====================================================

SELECT
    relocation_willingness,
    COUNT(*) AS total_candidates
FROM final_data
GROUP BY relocation_willingness;

-- =====================================================
-- COMPETITION LEVEL ANALYSIS
-- =====================================================

SELECT
    competition_level,
    COUNT(*) AS total_candidates
FROM final_data
GROUP BY competition_level;

-- =====================================================
-- AVERAGE EXPECTED CTC
-- =====================================================

SELECT
    AVG(expected_ctc_lpa) AS avg_expected_ctc
FROM final_data;

-- =====================================================
-- PLACEMENT SUCCESS BY SPECIALIZATION
-- =====================================================

SELECT
    degree_specialization,
    AVG(status) * 100 AS placement_rate
FROM final_data
GROUP BY degree_specialization
ORDER BY placement_rate DESC;

-- =====================================================
-- TOP 5 TECHNICAL SCORES
-- =====================================================

SELECT
    technical_score,
    communication_score,
    skills_match_percentage
FROM final_data
ORDER BY technical_score DESC
LIMIT 5;

-- =====================================================
-- FINAL INSIGHTS
-- =====================================================

/*

Key Insights:

1. Technical score strongly influences placement success.

2. Higher skills match percentage improves hiring probability.

3. Certain degree specializations show higher placement rates.

4. Experience level impacts job acceptance probability.

5. Interview performance categories help identify hiring trends.

6. SQL analytical queries support recruitment decision-making.

*/