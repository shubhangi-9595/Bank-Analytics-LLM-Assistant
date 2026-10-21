# Transactions and payment orders

## trans.type: transaction direction
trans.type is the direction of a transaction:
- 'PRIJEM' = credit (money in)
- 'VYDAJ' = withdrawal or debit (money out)
- 'VYBER' = withdrawal in cash (a smaller group of debits)

## trans.operation: transaction mode
trans.operation is how the transaction happened:
- 'VYBER KARTOU' = credit card withdrawal
- 'VKLAD' = credit in cash
- 'PREVOD Z UCTU' = collection from another bank
- 'VYBER' = withdrawal in cash
- 'PREVOD NA UCET' = remittance to another bank
Interest credited by the bank has an empty operation.

## trans.k_symbol: transaction purpose
trans.k_symbol is the characterization (purpose) of the transaction:
- 'POJISTNE' = insurance payment
- 'SLUZBY' = payment for statement
- 'UROK' = interest credited
- 'SANKC. UROK' = sanction interest charged because the balance was negative
- 'SIPO' = household payment
- 'DUCHOD' = old-age pension
- 'UVER' = loan payment
An empty string or a single space means the purpose was not recorded.

## trans.amounts: amount, balance and partner
trans.amount is the amount of the transaction (always positive; the direction is in trans.type).
trans.balance is the account balance after the transaction; a negative balance means the account is overdrawn.
trans.bank is the partner's bank as a two-letter code (e.g. 'AB'), trans.account the partner's account number.
trans.date is the transaction date.

## order.k_symbol: permanent order purpose
The table `order` holds permanent payment orders (standing debits). order.amount is the debited amount,
order.bank_to and order.account_to the recipient's bank and account. order.k_symbol is the purpose:
- 'POJISTNE' = insurance payment
- 'SIPO' = household payment
- 'LEASING' = leasing
- 'UVER' = loan payment
A single space means not specified.
