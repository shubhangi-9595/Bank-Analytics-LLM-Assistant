# Dataset overview

## overview.dataset: what this data is
Anonymised data from a real Czech bank, released for the PKDD'99 Discovery Challenge (often called the
"Berka" dataset). It covers 1993 to 1998: 4,500 accounts, 5,369 clients, 1,056,320 transactions,
682 loans, 892 credit cards, 6,471 permanent payment orders and 77 districts with demographic data.
All amounts are in Czech crowns (CZK). Dates are stored as text in the format YYYY-MM-DD, so a year
can be extracted with STRFTIME('%Y', date) and a month filtered with date LIKE '1996-01%'.

## overview.relationships: how the tables join
- client -> disp -> account: a client is linked to an account through the disposition table disp
  (disp.client_id, disp.account_id). One client can have several accounts and one account several clients.
- card -> disp: a credit card belongs to a disposition (card.disp_id = disp.disp_id), not directly to an account.
- loan -> account: loan.account_id. An account has at most one loan.
- trans -> account and order -> account: account_id. The table name "order" is a SQL keyword, so write `order` or "order".
- district: account.district_id is the branch where the account was opened; client.district_id is where the client lives.
  They can differ, so choose the one the question asks about.
