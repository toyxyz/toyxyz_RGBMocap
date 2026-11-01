"""Cache aliases for reviewed policy/logging-only changes.

Match complete file bytes: later numerical edits automatically invalidate caches.
Do not add aliases for changed inference algorithms or settings.
"""
from .storage import sha256

POLICY_ONLY = {
    "6d2702884592ca28a6fba2875b24dc2afd7ef149f8bfb24ac11acb95d9c60f0f": "8e5e7e5e2c6f677299d4a6081f20c5e7c2fb0fe17f18c0e0a4e5599bb01f8b1a",
    "61da72ab4b9ea248ee5c79eefee6282b91c7dc1773fcf72e9186d382893c4f04": "8e5e7e5e2c6f677299d4a6081f20c5e7c2fb0fe17f18c0e0a4e5599bb01f8b1a",
    "a0fa66d293dde00400c3367fc68cb6b9faeae591c87cabeeb8e06a0bf9f5feaa": "d0d32032fad3470cc4801134d4bf4fafdf8bfc5bc805228ece5cc8dabe4f370e",
    "0b479a3a33a4f4df3804ee57e233f314dd5554044c1af6a274317878241fc972": "6c1de70d9628ae41528a4494de39034764e0b938982d61c7470e631a371f4435",
    "f47c0da4b7f7671931174e2aaf8ed09a7d1fde1047fd3c07d96065b8cc14425b": "3125b5a360d71c229a958a553548af0f03c82b58698f1b758a14f2048a2f7e92",
    "7cf59b6f095e76bbaf372664659a0fe4810a499473e06db64056cf4865bdd4d6": "8e5e7e5e2c6f677299d4a6081f20c5e7c2fb0fe17f18c0e0a4e5599bb01f8b1a",
    "951f52af4e4820ab63cda224dbda7b6e3d464a3d893a81759d89b99226546e1e": "6c1de70d9628ae41528a4494de39034764e0b938982d61c7470e631a371f4435",
    "a2fb18686236d6852df13f03b95890860e961b63072d809eeaf5cd3331edd6bc": "3125b5a360d71c229a958a553548af0f03c82b58698f1b758a14f2048a2f7e92",
    "3d166c1b0807397328f82f973b7617118b3c35d6435f4f85036bd8be2925268a": "d0d32032fad3470cc4801134d4bf4fafdf8bfc5bc805228ece5cc8dabe4f370e",
    "86e0f23ca7a6d06bf177436bf8d2a8462577a8f151d3bdc7380c63dcdef016a3": "8e5e7e5e2c6f677299d4a6081f20c5e7c2fb0fe17f18c0e0a4e5599bb01f8b1a"
}


# UTF-8 log streams change no SAM features. Complete-byte matching means
# any subsequent numerical edit still invalidates the affected cache.
LOGGING_ONLY = {
    "db6fd57ca78c6dfab27142b9e99db207fcc1021933f1c4cfeb599c5fdad321ed": "c1abd10ec3143cfa7d7b97283a461fa858b73f55338cde7d7842cf030eff0b95",
    "141809da55236c2cbbf3f26f9e925362bcac6197d3883406347fa6b3ac7b7174": "319f211033a21d0fa1a83dd01a45676844df93cf95680e0d9fb7242883e3ba42"
}


def code_hash(path):
    actual=sha256(path)
    return LOGGING_ONLY.get(actual,POLICY_ONLY.get(actual,actual))
