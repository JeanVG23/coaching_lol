from fastapi import APIRouter

import readers

router = APIRouter()


@router.get("/api/accounts")
def list_accounts():
    return readers.account_summaries()