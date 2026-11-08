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


# Download callback plumbing changes no MAMMA inference inputs.
# These aliases match complete reviewed bytes, never arbitrary later edits.
DOWNLOAD_PROGRESS_ONLY = {
    "c3b12bed270ffea56272e750b234e06c6c49ce7eea980d0520dd0c8180c517aa": "e98a7df6d9396ee5d8ef9411b76cf899497f6c873955812b18b7439376da2e0d",
    "8252ef60d906f1c36a8d3b832fbdf9da8204bedf66d3adecb55de52580121413": "5c90318cfc520bfad4f84d0239a47bdc38b95be14f2f3fc79baf9d9c7e9f32ba"
}


# Cross-platform plumbing (interpreter paths, process/link helpers, platform
# suffixes) changes no inference inputs; Windows retains its exact behavior.
# Complete-byte matching: any later numerical edit still invalidates caches.
PLATFORM_ONLY = {
    "eef97f92a58fd40432159bd19a201f0f9c6f1579959b18f776e55c2af754b175": "883d9dde808cf27c86611cb85f0e28ffcb3541639d417d2f68c4c7e1b2d00855",  # worker.py
    "b6d91ac3bedd245c6be83ac821a6cf87d06e34fd16ed0cd27f6c67a2c4d6ba97": "74512a9b2fb87f4fafb9f0bc963f5b5887e4069ffc59c24f656f984b9c8066fd",  # engines.py
    "18b52409527dea9e7513850f7936d8662f09c93f00d64d8452b449881b2af1b8": "7b25e1eb57ed810bdf995a918d7b7633e7e558c25f982ba273f7d64783105ec3",  # correction.py
    "4f6711a1d08835c2e3894a7780a89432ef5f84a6259ce094368d344ce8cfc3e3": "7d1061f7f56e6c47a7a9660979616fb751dd8da36f39f0f4ed72a3b81135e04d",  # handflow_engine.py
    "bd97d0328b539a60af2b135ccb9d7ec669d40e71c7509274625d8bfc75a11795": "f157fe40e4c65b4853425ea50ed641a3b1a91fde40225aae93e273f912b05568",  # handflow.py
    "14dfa2e611eb14335b38619469860b0c11baa7666ba8cee5e86636b1ed96bfac": "77c7162093f5bbf9b4dba46287f6542d5069cd3aef24444f531eac94335bd82f",  # exmr_backend.py
    "c851aa0b51b55b72840f2d4bf9afa9d2c50f2549a16802d4ebbb01e28eafca11": "b82861ea8496949c3b7d914a4b6eecf6400e61dcc008a188e2b7504c317a3012",  # flow_hamer.py
    "f0192f642b91e09cd4d5061134bdc07b6f835e57ce3d74f66ce974ff090d5c9e": "6fb33f433ece03ef19099a86d803a31a2b67ac23c70aa95a892906afc20e4d7a",  # flow_correction.py
    "97aad245e37c7ffa7792f777c39d9423aa75f62aac00254733e25be3cc58baef": "2e4533bdd021f57cc01baeb4c36ffe396d68ca512b743a5fd4d2cad158af65e7",  # gvhmr_preprocess.py
    "4474028fcc8f609f01feca20d976a26826d6db217a98b6010bc1eed08fcb73c1": "5c90318cfc520bfad4f84d0239a47bdc38b95be14f2f3fc79baf9d9c7e9f32ba",  # mamma_pipeline.py
    "8d62a79ede7eb2fdc8cb0e3e1d312c9b74d2e9f318875a268f4bf5a543887fc0": "68ae06b6c230612066591c2380506dae6886b2ccb779d79c938ff14cb9894b0d",  # runtime_registry.py
    "df87cb443c3c148e511263781d3640df889253eae3f751cc10126cd8e6c1ccaa": "e3d094bbc21a7af4aa64dc857f004d7d15fdf931e163ec137ce53393301181ba",  # flow_backend adapter: flow_camera
    "8edfd3f96c55f42859e32b21606c85a8c37028a1e63654a8d240963b86d1214f": "6d9081574acbb859153820cc8d5c57676d15c70261043d7b0035a395875e6ac2",  # flow_backend adapter: flow_features
    "fd67ebadffd8635eed91500effe2cb74c5bbb820ba3865770d76fc8e519df35e": "11164f85046c4548f58bc74298b34e1f01b6b4b33309b314f4307d7a37b72a1b"  # flow_backend adapter: flowhmr
}


def alias(digest):
    """Identity value of reviewed source bytes; any other digest is its own identity."""
    return PLATFORM_ONLY.get(digest,DOWNLOAD_PROGRESS_ONLY.get(digest,LOGGING_ONLY.get(digest,POLICY_ONLY.get(digest,digest))))


def code_hash(path):
    return alias(sha256(path))
