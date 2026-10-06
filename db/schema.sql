-- CAMS database schema. MySQL 5.7 or newer (generated columns need 5.7), InnoDB, utf8mb4.
-- Apply with: python -m db.init_db
--
-- Keep one column or constraint per line and no semicolons except between statements:
-- db/__init__.py and tests/test_schema.py read this file line by line.
--
-- NO DOUBLE-BOOKING IS ENFORCED HERE, NOT IN PYTHON (CLAUDE.md hard rule 1):
--   1. slots holds exactly one row per doctor, date and time (UNIQUE).
--   2. bookings.confirmed_slot_id equals slot_id for every status except 'cancelled', where it is NULL.
--      Its UNIQUE key allows one active booking per slot (completed and no_show keep the slot)
--      and any number of cancelled ones, because UNIQUE ignores NULLs.
--   3. The app just INSERTs a booking and treats a duplicate-key error as "slot taken" (HTTP 409).
--
-- Foreign keys keep the default RESTRICT on purpose: MySQL forbids CASCADE and SET NULL on the
-- base column (slot_id) of a stored generated column.

CREATE TABLE users (
  id            INT          NOT NULL AUTO_INCREMENT,
  name          VARCHAR(120) NOT NULL,
  email         VARCHAR(255) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  role          ENUM('patient','doctor','admin') NOT NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_users_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE patient_profiles (
  id            INT          NOT NULL AUTO_INCREMENT,
  user_id       INT          NOT NULL,
  date_of_birth DATE         NOT NULL,
  sex           ENUM('F','M') NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_patient_profiles_user (user_id),
  CONSTRAINT fk_patient_profiles_user FOREIGN KEY (user_id) REFERENCES users (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE doctors (
  id             INT          NOT NULL AUTO_INCREMENT,
  user_id        INT          NOT NULL,
  specialization VARCHAR(100) NOT NULL,
  slot_minutes   INT          NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_doctors_user (user_id),
  CONSTRAINT fk_doctors_user FOREIGN KEY (user_id) REFERENCES users (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE slots (
  id        INT  NOT NULL AUTO_INCREMENT,
  doctor_id INT  NOT NULL,
  slot_date DATE NOT NULL,
  slot_time TIME NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_slots_doctor_date_time (doctor_id, slot_date, slot_time),
  CONSTRAINT fk_slots_doctor FOREIGN KEY (doctor_id) REFERENCES doctors (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE bookings (
  id                INT          NOT NULL AUTO_INCREMENT,
  slot_id           INT          NOT NULL,
  patient_id        INT          NOT NULL,
  status            ENUM('confirmed','completed','no_show','cancelled') NOT NULL DEFAULT 'confirmed',
  reason            VARCHAR(500) NULL,
  created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  cancelled_at      DATETIME     NULL,
  cancelled_by      INT          NULL,
  confirmed_slot_id INT GENERATED ALWAYS AS (CASE WHEN status = 'cancelled' THEN NULL ELSE slot_id END) STORED,
  PRIMARY KEY (id),
  UNIQUE KEY uq_bookings_confirmed_slot (confirmed_slot_id),
  CONSTRAINT fk_bookings_slot FOREIGN KEY (slot_id) REFERENCES slots (id),
  CONSTRAINT fk_bookings_patient FOREIGN KEY (patient_id) REFERENCES users (id),
  CONSTRAINT fk_bookings_cancelled_by FOREIGN KEY (cancelled_by) REFERENCES users (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
