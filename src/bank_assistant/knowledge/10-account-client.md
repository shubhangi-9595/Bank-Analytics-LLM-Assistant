# Accounts, clients and dispositions

## account.frequency: statement issuance frequency
account.frequency says how often statements are issued for the account:
- 'POPLATEK MESICNE' = monthly issuance
- 'POPLATEK TYDNE' = weekly issuance
- 'POPLATEK PO OBRATU' = issuance after each transaction
account.date is the date the account was opened.

## client.gender_birth: client gender and birth date
client.gender is the client's gender:
- 'M' = male
- 'F' = female
client.birth_date is the date of birth (YYYY-MM-DD).
In the raw data both were packed into one "birth number" (YYMMDD, with 50 added to the month for women).
An earlier birth date means an older client. Age at a given date = difference in years between that date and birth_date.

## disp.type: disposition type (owner or user)
disp.type is the client's rights on the account:
- 'OWNER' = the account owner. Only owners can issue permanent orders and ask for a loan.
- 'DISPONENT' = a user with limited rights on someone else's account.
Every account has exactly one OWNER.
