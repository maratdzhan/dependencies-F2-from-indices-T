# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(7000)

block_cipher = None


a = Analysis(['F2P.py'],
             pathex=['D:\\Jupyter\\work\\Profi\\Ulia\\dependencies-F2-from-indices-T'],
             binaries=[],
             datas=[],
             hiddenimports=['pandas'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='F2P',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='F2P')
