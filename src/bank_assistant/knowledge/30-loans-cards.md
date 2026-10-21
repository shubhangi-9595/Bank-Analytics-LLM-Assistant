# Loans and credit cards

## loan.status: loan repayment status
loan.status is the state of the loan contract:
- 'A' = contract finished, no problems
- 'B' = contract finished, loan not paid
- 'C' = running contract, OK so far
- 'D' = running contract, client in debt
Finished contracts are A and B; running contracts are C and D.

## loan.terms: loan amount, duration and payments
loan.amount is the amount of the loan, loan.duration its length in months (12, 24, 36, 48 or 60),
loan.payments the monthly payment, loan.date the date the loan was granted.

## card.type: credit card type
card.type is the kind of card:
- 'junior' = junior card (entry level)
- 'classic' = classic card
- 'gold' = gold card (the high-level card)
card.issued is the date the card was issued. A card links to a client through disp (card.disp_id).

## metric.default: what counts as a loan default (project definition)
In this project a loan has DEFAULTED if its status is 'B' (finished, not paid) or 'D' (running, client in debt).
Default rate = defaulted loans / all loans. Status 'C' loans may still default later, so the label is
imperfect for recent loans.
