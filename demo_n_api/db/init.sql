DROP DATABASE ans;
CREATE DATABASE ans;
USE ans;


CREATE TABLE IF NOT EXISTS accounts (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100),
    password VARCHAR(100),
    email VARCHAR(50),
    role VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS suggestions (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    eng TEXT,
    suggestion TEXT,
    trans TEXT,
    user_type VARCHAR(50)
);

INSERT INTO accounts
  (username, password, email, role)
VALUES
  ('kskianis', '$2b$12$EUbIcyX1R9gvylXkLcStFO54a4FZLQk6ovUF8yuDUfgXrzxvwgKU2', 'rob.cs.aueb@gmail.com', 'admin'),
  ('tayeb83', '$2b$12$DAVKsB2i8/XsbILCo8x2Pe97Kq2xvOl4tZ7gds957hcQ.pccpa5Tm', 'tayeb.merabti@gmail.com', 'expert'),
  ('melissa', '$2b$12$l2pgLTigayQD4DUuRR.fQOk5pVil0xYNtKjdLqL4koKUHIghnMY46', 'melissa.mary@esante.gouv.fr', 'expert'),
  ('ans_interop_synt', '$2b$12$DZXZQMQLTVc7Oc.D9qsAf.TbQGaWidOpkRXHDD71MSPdsDbBBfL5C', 'rob.cs.aueb@gmail.com', 'user'),
  ('ans_interop_sem', '$2b$12$sAs.5A8K3FzveA/N0/dNJ.K0ua72bv/h0Xi.zwe8gXHIAVlVJUZKy', 'rob.cs.aueb@gmail.com', 'user');