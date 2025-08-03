#!/bin/bash
# This script creates the database and user, and sets the permissions for Tirtha.
# For environment variables, see the tirtha.env file.

set -e

sudo su - postgres -c \
"psql <<__END__

CREATE DATABASE $DB_NAME;
CREATE USER $DB_USER WITH PASSWORD '$DB_PWD';
ALTER ROLE $DB_USER SET client_encoding TO 'utf8';
ALTER ROLE $DB_USER SET default_transaction_isolation TO 'read committed';
ALTER ROLE $DB_USER SET timezone TO 'UTC';
ALTER DATABASE $DB_NAME OWNER TO $DB_USER;
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
GRANT CREATE ON SCHEMA public TO $DB_USER;

__END__
"
