# -*- coding: utf-8 -*-

"""
===============================================================================
ETAPA 04
CONDIÇÃO HIDROLÓGICA + REDE HIDROGRÁFICA + SAÚDE
===============================================================================

Integra:

- SipamHidro
- série histórica / climatologia das cotas
- BHO250 - trechos de drenagem
- municípios
- Polos Base
- UBS
- UBSI
- UPA

LÓGICA ESPACIAL
---------------
Existe apenas UM raio de análise.

Exemplo:
Raio = 50 km

Então o painel mostra, dentro dos mesmos 50 km:

- trechos de drenagem;
- rede hidrográfica conectada à estação;
- municípios;
- Polos Base;
- UBS;
- UBSI;
- UPA.

Também calcula a distância das estruturas até a estação.

IMPORTANTE
----------
A condição hidrológica apresentada refere-se à estação selecionada.

A visualização dos cursos d'água próximos não significa que todos eles
apresentem a mesma condição observada na estação.

A proximidade de uma estrutura de saúde também não representa, por si só,
exposição, inundação, isolamento ou risco epidemiológico.

===============================================================================
"""

from pathlib import Path
import os
import base64
from functools import lru_cache
import math
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    dash_table,
)

import plotly.graph_objects as go


warnings.filterwarnings("ignore")


# =============================================================================
# CAMINHOS - PORTÁVEIS (GITHUB / RENDER / LOCAL)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

PASTA_DADOS = BASE_DIR / "dados"
PASTA_DADOS_ANA = BASE_DIR / "dados_ana"

ARQ_ESTACOES = PASTA_DADOS / "estacoes_atual.parquet"
ARQ_HISTORICO = PASTA_DADOS / "cotas_historico.parquet"

ARQ_ANA_ATUAL = PASTA_DADOS_ANA / "ana_estacoes_com_cota_atual.parquet"
ARQ_ANA_REFERENCIA_MENSAL = PASTA_DADOS_ANA / "ana_referencia_mensal.parquet"

ARQ_POLOS = BASE_DIR / "polos_base.json"

# BHO250 não é enviado ao Render nesta primeira versão.
# Se o arquivo for adicionado futuramente à raiz, o painel volta a utilizá-lo.
ARQ_BHO = BASE_DIR / "geoft_bho_trecho_drenagem.gpkg"
CAMADA_BHO = "pgh_output.geoft_bho_trecho_drenagem"
BHO_DISPONIVEL = ARQ_BHO.exists()


def localizar_geojson(nome):
    arquivo = BASE_DIR / nome
    if arquivo.exists():
        return arquivo
    raise FileNotFoundError(f"\nArquivo não encontrado: {arquivo}")


ARQ_MUNICIPIOS = localizar_geojson("municipios_br.geojson")
ARQ_UBS = localizar_geojson("ubs.geojson")
ARQ_UBSI = localizar_geojson("ubsi.geojson")
ARQ_UPA = localizar_geojson("upa.geojson")


# =============================================================================
# SERVIDOR
# =============================================================================

HOST = "0.0.0.0"

PORT = int(os.environ.get("PORT", "8067"))


# =============================================================================
# CRS
# =============================================================================

CRS_GEO = 4326

CRS_BHO = 4674

CRS_METRICO = 5880


# =============================================================================
# SEGURANÇA DA REDE
# =============================================================================

MAX_TRECHOS_CONECTADOS = 15000


# =============================================================================
# CORES
# =============================================================================

CORES_CONDICAO = {
    "INUNDAÇÃO": "#7f0000",
    "ALERTA — CHEIA": "#b30000",
    "ATENÇÃO — CHEIA": "#ef6548",
    "ATENÇÃO — ESTIAGEM": "#b30000",
    "MUITO ABAIXO DO ESPERADO": "#b30000",
    "MUITO ACIMA DO ESPERADO": "#006837",
    "ABAIXO DO ESPERADO": "#ef6548",
    "ACIMA DO ESPERADO": "#31a354",
    "DENTRO DO ESPERADO": "#7a7a7a",
    "SEM CLASSIFICAÇÃO": "#bdbdbd",
}

def estilo_estacao(condicao, tendencia=None):

    c = str(condicao).upper().strip()

    # Sem classificação
    if c == "SEM CLASSIFICAÇÃO":
        return "triangle-up", "#bdbdbd"

    # Condições de baixa cota / estiagem
    if "MUITO ABAIXO DO ESPERADO" in c or "ESTIAGEM" in c:
        return "triangle-down", "#b30000"

    if "ABAIXO DO ESPERADO" in c:
        return "triangle-down", "#ef6548"

    # Condições de cota elevada
    if "MUITO ACIMA DO ESPERADO" in c:
        return "triangle-up", "#006837"

    if "ACIMA DO ESPERADO" in c:
        return "triangle-up", "#31a354"

    # Dentro da faixa esperada
    if c == "DENTRO DO ESPERADO":
        return "triangle-up", "#7a7a7a"

    # Condições oficiais relacionadas à cheia
    if "INUNDAÇÃO" in c:
        return "triangle-up", "#7f0000"

    if "ALERTA — CHEIA" in c:
        return "triangle-up", "#b30000"

    if "ATENÇÃO — CHEIA" in c:
        return "triangle-up", "#ef6548"

    return "triangle-up", "#bdbdbd"



# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def normalizar_codigo(serie):

    return (
        serie
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.strip()
    )


def fmt_numero(
    valor,
    casas=2,
):

    if pd.isna(valor):

        return "—"

    return (
        f"{float(valor):.{casas}f}"
        .replace(
            ".",
            ","
        )
    )


def fmt_data(valor):

    if pd.isna(valor):

        return "—"

    valor = pd.Timestamp(
        valor
    )

    if valor.tzinfo is not None:

        valor = valor.tz_convert(
            None
        )

    return valor.strftime(
        "%d/%m/%Y %H:%M"
    )


def texto_valido(valor):

    if pd.isna(valor):

        return ""

    valor = str(
        valor
    ).strip()

    if valor.lower() in [
        "",
        "nan",
        "none",
    ]:

        return ""

    return valor


def primeiro_campo(
    colunas,
    candidatos,
):

    mapa = {

        str(c)
        .lower()
        .strip():
            c

        for c
        in colunas
    }

    for candidato in candidatos:

        chave = (
            candidato
            .lower()
            .strip()
        )

        if chave in mapa:

            return mapa[
                chave
            ]

    return None


def chave_no(valor):

    """
    Normaliza noorigem/nodestino para comparação topológica.
    """

    if pd.isna(valor):

        return None

    try:

        numero = float(
            valor
        )

        if numero.is_integer():

            return str(
                int(
                    numero
                )
            )

    except Exception:

        pass

    return str(
        valor
    ).strip()


# =============================================================================
# CARD
# =============================================================================

def criar_card(
    titulo,
    valor,
    subtitulo=None,
    cor_borda=None,
):

    filhos = [

        html.Div(

            titulo,

            style={
                "fontSize":
                    "13px",

                "color":
                    "#666",

                "marginBottom":
                    "5px",
            },
        ),

        html.Div(

            valor,

            style={
                "fontSize":
                    "23px",

                "fontWeight":
                    "bold",
            },
        ),
    ]


    if subtitulo:

        filhos.append(

            html.Div(

                subtitulo,

                style={
                    "fontSize":
                        "11px",

                    "color":
                        "#777",

                    "marginTop":
                        "5px",
                },
            )
        )


    estilo = {

        "backgroundColor":
            "white",

        "padding":
            "14px",

        "borderRadius":
            "8px",

        "boxShadow":
            "0 1px 4px rgba(0,0,0,0.10)",
    }


    if cor_borda:

        estilo[
            "borderLeft"
        ] = (
            f"6px solid {cor_borda}"
        )


    return html.Div(
        filhos,
        style=estilo,
    )


# =============================================================================
# GEOMETRIAS PARA PLOTLY
# =============================================================================

def geometrias_para_linhas(
    gdf,
):

    lons = []

    lats = []


    for geom in gdf.geometry:

        if (
            geom is None
            or
            geom.is_empty
        ):

            continue


        if geom.geom_type == "Polygon":

            for lon, lat in (
                geom
                .exterior
                .coords
            ):

                lons.append(
                    lon
                )

                lats.append(
                    lat
                )

            lons.append(
                None
            )

            lats.append(
                None
            )


        elif geom.geom_type == "MultiPolygon":

            for pol in geom.geoms:

                for lon, lat in (
                    pol
                    .exterior
                    .coords
                ):

                    lons.append(
                        lon
                    )

                    lats.append(
                        lat
                    )

                lons.append(
                    None
                )

                lats.append(
                    None
                )


        elif geom.geom_type == "LineString":

            for lon, lat in geom.coords:

                lons.append(
                    lon
                )

                lats.append(
                    lat
                )

            lons.append(
                None
            )

            lats.append(
                None
            )


        elif geom.geom_type == "MultiLineString":

            for linha in geom.geoms:

                for lon, lat in linha.coords:

                    lons.append(
                        lon
                    )

                    lats.append(
                        lat
                    )

                lons.append(
                    None
                )

                lats.append(
                    None
                )


    return (
        lons,
        lats,
    )


# =============================================================================
# NOME DO CURSO D'ÁGUA
# =============================================================================

def obter_nome_rio(row):

    for campo in [

        "noriocomp",
        "noespecif",
        "nooriginal",
    ]:

        if campo in row.index:

            valor = texto_valido(
                row[
                    campo
                ]
            )

            if valor:

                return valor


    return "Curso d'água sem nome"


def obter_nome_bacia(row):

    for campo in [
        "nobacia",
        "no_bacia",
        "nomebacia",
        "nome_bacia",
        "nmbacia",
        "nm_bacia",
    ]:

        if campo in row.index:

            valor = texto_valido(
                row[
                    campo
                ]
            )

            if valor:

                return valor

    return None


# =============================================================================
# INÍCIO
# =============================================================================

print(
    "\n"
    +
    "#" * 80
)

print(
    "ETAPA 04"
)

print(
    "CONDIÇÃO HIDROLÓGICA + REDE HIDROGRÁFICA + SAÚDE"
)

print(
    "#" * 80
)



# =============================================================================
# PREVISÃO SAZONAL DE VAZÃO - CEMADEN - SETEMBRO/2026
# =============================================================================

ARQ_PREVISAO_CEMADEN = BASE_DIR / "previsao_hidrologia_cemaden_setembro.png"


def arquivo_para_data_uri(caminho):

    caminho = Path(caminho)

    if not caminho.exists():

        return None

    conteudo = base64.b64encode(
        caminho.read_bytes()
    ).decode(
        "ascii"
    )

    return (
        "data:image/png;base64,"
        +
        conteudo
    )


PREVISAO_CEMADEN_IMG = arquivo_para_data_uri(
    ARQ_PREVISAO_CEMADEN
)

PREVISAO_CEMADEN_RESUMO = (
    "A previsão mensal foi obtida por meio do site de monitoramento hidrológico "
    "do CEMADEN. Para setembro de 2026, a previsão sazonal de vazões naturais "
    "do Sistema GloFAS indica contrastes entre as regiões do Brasil. No Norte, "
    "há vazões pouco acima da média no oeste da Amazônia, especialmente na bacia "
    "do rio Juruá, enquanto áreas das bacias dos rios Negro, Branco, Araguari, "
    "Tapajós, Xingu e Tocantins-Araguaia apresentam redução das vazões. No Nordeste, "
    "destaca-se forte redução entre Maranhão e Piauí, com condições menos severas "
    "próximo ao litoral. No Centro-Oeste e em partes do Sudeste, predominam vazões "
    "abaixo da média, enquanto na Região Sul prevalecem condições próximas ou "
    "levemente acima da média. De forma geral, a previsão aponta redução dos "
    "escoamentos no Brasil central e no interior do Nordeste, em contraste com "
    "vazões mais elevadas no oeste da Amazônia e em áreas do Sul."
)

# Interpretação visual preliminar da figura, usada apenas neste protótipo.
# Não substitui dado vetorial/grade oficial do Cemaden.
UFS_PREVISAO_ATENCAO = {
    "RR", "AM", "PA", "AP", "MT", "TO", "GO", "DF",
    "MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"
}

# =============================================================================
# MUNICÍPIOS
# =============================================================================

print(
    "\nLendo municípios..."
)


MUN = gpd.read_file(
    ARQ_MUNICIPIOS
)


if MUN.crs is None:

    MUN = MUN.set_crs(
        CRS_GEO
    )

else:

    MUN = MUN.to_crs(
        CRS_GEO
    )


CAMPO_NOME_MUN = primeiro_campo(

    MUN.columns,

    [
        "NM_MUN",
        "nome",
        "municipio",
        "NM_MUNICIP",
    ],
)


CAMPO_UF_MUN = primeiro_campo(

    MUN.columns,

    [
        "SIGLA_UF",
        "SG_UF",
        "UF",
    ],
)


CAMPO_COD_MUN = primeiro_campo(

    MUN.columns,

    [
        "CD_MUN",
        "COD_MUN",
        "codigo",
    ],
)


MUN[
    "mun_nome"
] = (

    MUN[
        CAMPO_NOME_MUN
    ]
    .fillna("")
    .astype(str)

    if CAMPO_NOME_MUN

    else ""
)


MUN[
    "mun_uf"
] = (

    MUN[
        CAMPO_UF_MUN
    ]
    .fillna("")
    .astype(str)

    if CAMPO_UF_MUN

    else ""
)


MUN[
    "mun_codigo"
] = (

    MUN[
        CAMPO_COD_MUN
    ]
    .fillna("")
    .astype(str)

    if CAMPO_COD_MUN

    else ""
)


MUN_JOIN = MUN[
    [
        "mun_nome",
        "mun_uf",
        "mun_codigo",
        "geometry",
    ]
].copy()


MUN_PROJ = MUN.to_crs(
    CRS_METRICO
)


# =============================================================================
# POLOS BASE
# =============================================================================

print(
    "Lendo Polos Base..."
)


POLOS = gpd.read_file(
    ARQ_POLOS
)


if POLOS.crs is None:

    raise ValueError(
        "Shapefile dos Polos Base está sem CRS."
    )


POLOS = POLOS.to_crs(
    CRS_GEO
)


CAMPO_POLO = primeiro_campo(

    POLOS.columns,

    [
        "polo",
        "DS_POLO_BA",
        "nome_polo",
    ],
)


CAMPO_DSEI = primeiro_campo(

    POLOS.columns,

    [
        "dsei",
        "DSEI_GESTA",
        "DS_DSEI",
    ],
)


CAMPO_UF_POLO = primeiro_campo(

    POLOS.columns,

    [
        "UF",
        "SG_UF",
    ],
)


POLOS[
    "polo_nome"
] = (

    POLOS[
        CAMPO_POLO
    ]
    .fillna("")
    .astype(str)

    if CAMPO_POLO

    else ""
)


POLOS[
    "dsei_nome"
] = (

    POLOS[
        CAMPO_DSEI
    ]
    .fillna("")
    .astype(str)

    if CAMPO_DSEI

    else ""
)


POLOS[
    "polo_uf"
] = (

    POLOS[
        CAMPO_UF_POLO
    ]
    .fillna("")
    .astype(str)

    if CAMPO_UF_POLO

    else ""
)


POLOS_JOIN = POLOS[
    [
        "polo_nome",
        "dsei_nome",
        "polo_uf",
        "geometry",
    ]
].copy()


POLOS_PROJ = POLOS.to_crs(
    CRS_METRICO
)


# =============================================================================
# ENRIQUECER PONTOS
# =============================================================================

def enriquecer_pontos(
    gdf,
):

    gdf = gdf.copy()


    if "_id" not in gdf.columns:

        gdf[
            "_id"
        ] = np.arange(
            len(
                gdf
            )
        )


    # MUNICÍPIO
    temp = gpd.sjoin(

        gdf,

        MUN_JOIN,

        how=
            "left",

        predicate=
            "within",
    )


    temp = (
        temp
        .sort_values(
            "_id"
        )
        .drop_duplicates(
            "_id"
        )
    )


    if "index_right" in temp.columns:

        temp = temp.drop(
            columns=[
                "index_right"
            ]
        )


    # POLO
    temp = gpd.sjoin(

        temp,

        POLOS_JOIN,

        how=
            "left",

        predicate=
            "within",
    )


    temp = (
        temp
        .sort_values(
            "_id"
        )
        .drop_duplicates(
            "_id"
        )
    )


    if "index_right" in temp.columns:

        temp = temp.drop(
            columns=[
                "index_right"
            ]
        )


    return temp


# =============================================================================
# UBS
# =============================================================================

print(
    "Lendo UBS..."
)


ubs = gpd.read_file(
    ARQ_UBS
)


if ubs.crs is None:

    ubs = ubs.set_crs(
        4674
    )


ubs = ubs.to_crs(
    CRS_GEO
)


ubs[
    "_id"
] = (
    "UBS_"
    +
    ubs.index.astype(
        str
    )
)


ubs[
    "tipo"
] = "UBS"


ubs[
    "nome"
] = (
    ubs[
        "no_fantasi"
    ]
    .fillna(
        "UBS"
    )
    .astype(str)
)


ubs[
    "cnes"
] = (
    ubs[
        "co_cnes"
    ]
    .fillna("")
    .astype(str)
    .str.replace(
        ".0",
        "",
        regex=False,
    )
    .str.zfill(
        7
    )
)


ubs = enriquecer_pontos(
    ubs
)


ubs[
    "municipio"
] = ubs[
    "mun_nome"
].fillna("")


ubs[
    "uf"
] = ubs[
    "mun_uf"
].fillna("")


ubs[
    "polo_base"
] = ubs[
    "polo_nome"
].fillna("")


ubs[
    "dsei"
] = ubs[
    "dsei_nome"
].fillna("")


# =============================================================================
# UBSI
# =============================================================================

print(
    "Lendo UBSI..."
)


ubsi = gpd.read_file(
    ARQ_UBSI
)


if ubsi.crs is None:

    ubsi = ubsi.set_crs(
        4674
    )


ubsi = ubsi.to_crs(
    CRS_GEO
)


ubsi[
    "_id"
] = (
    "UBSI_"
    +
    ubsi.index.astype(
        str
    )
)


ubsi[
    "tipo"
] = "UBSI"


ubsi[
    "nome"
] = (
    ubsi[
        "nome_da_es"
    ]
    .fillna(
        "UBSI"
    )
    .astype(str)
)


ubsi[
    "cnes"
] = ""


ubsi[
    "municipio"
] = (
    ubsi[
        "municipio_"
    ]
    .fillna("")
    .astype(str)
)


ubsi[
    "uf"
] = (
    ubsi[
        "uf_ibge"
    ]
    .fillna("")
    .astype(str)
)


ubsi[
    "polo_base"
] = (
    ubsi[
        "polo_base"
    ]
    .fillna("")
    .astype(str)
)


ubsi[
    "dsei"
] = (
    ubsi[
        "dsei"
    ]
    .fillna("")
    .astype(str)
)


ubsi = enriquecer_pontos(
    ubsi
)


ubsi[
    "municipio"
] = np.where(

    ubsi[
        "municipio"
    ].str.strip()
    !=
    "",

    ubsi[
        "municipio"
    ],

    ubsi[
        "mun_nome"
    ].fillna(""),
)


ubsi[
    "uf"
] = np.where(

    ubsi[
        "uf"
    ].str.strip()
    !=
    "",

    ubsi[
        "uf"
    ],

    ubsi[
        "mun_uf"
    ].fillna(""),
)


ubsi[
    "polo_base"
] = np.where(

    ubsi[
        "polo_base"
    ].str.strip()
    !=
    "",

    ubsi[
        "polo_base"
    ],

    ubsi[
        "polo_nome"
    ].fillna(""),
)


ubsi[
    "dsei"
] = np.where(

    ubsi[
        "dsei"
    ].str.strip()
    !=
    "",

    ubsi[
        "dsei"
    ],

    ubsi[
        "dsei_nome"
    ].fillna(""),
)


# =============================================================================
# UPA
# =============================================================================

print(
    "Lendo UPA..."
)


upa = gpd.read_file(
    ARQ_UPA
)


if upa.crs is None:

    upa = upa.set_crs(
        4674
    )


upa = upa.to_crs(
    CRS_GEO
)


upa[
    "_id"
] = (
    "UPA_"
    +
    upa.index.astype(
        str
    )
)


upa[
    "tipo"
] = "UPA"


upa[
    "nome"
] = (
    upa[
        "Nome Fanta"
    ]
    .fillna(
        "UPA"
    )
    .astype(str)
)


upa[
    "cnes"
] = (
    upa[
        "CNES"
    ]
    .fillna("")
    .astype(str)
    .str.replace(
        ".0",
        "",
        regex=False,
    )
    .str.zfill(
        7
    )
)


upa = enriquecer_pontos(
    upa
)


upa[
    "municipio"
] = upa[
    "mun_nome"
].fillna("")


upa[
    "uf"
] = upa[
    "mun_uf"
].fillna("")


upa[
    "polo_base"
] = upa[
    "polo_nome"
].fillna("")


upa[
    "dsei"
] = upa[
    "dsei_nome"
].fillna("")


# =============================================================================
# SAÚDE
# =============================================================================

COLUNAS_SAUDE = [

    "_id",
    "tipo",
    "nome",
    "cnes",
    "municipio",
    "uf",
    "polo_base",
    "dsei",
    "geometry",
]


SAUDE = pd.concat(

    [

        ubs[
            COLUNAS_SAUDE
        ],

        ubsi[
            COLUNAS_SAUDE
        ],

        upa[
            COLUNAS_SAUDE
        ],
    ],

    ignore_index=
        True,
)


SAUDE = gpd.GeoDataFrame(

    SAUDE,

    geometry=
        "geometry",

    crs=
        CRS_GEO,
)


SAUDE_PROJ = SAUDE.to_crs(
    CRS_METRICO
)


SINDEX_SAUDE = SAUDE_PROJ.sindex


# =============================================================================
# SIPAMHIDRO
# =============================================================================

print(
    "Lendo SipamHidro..."
)


estacoes = pd.read_parquet(
    ARQ_ESTACOES
)


historico = pd.read_parquet(
    ARQ_HISTORICO
)


estacoes[
    "codigo"
] = normalizar_codigo(
    estacoes[
        "codigo"
    ]
)


historico[
    "codigo"
] = normalizar_codigo(
    historico[
        "codigo"
    ]
)


historico[
    "data"
] = pd.to_datetime(

    historico[
        "data"
    ],

    errors=
        "coerce",
)


historico = historico[
    historico[
        "data"
    ].notna()
].copy()


estacoes[
    "latitude"
] = pd.to_numeric(

    estacoes[
        "latitude"
    ],

    errors=
        "coerce",
)


estacoes[
    "longitude"
] = pd.to_numeric(

    estacoes[
        "longitude"
    ],

    errors=
        "coerce",
)


if (
    "dataHoraUltimaMedicao"
    in estacoes.columns
):

    estacoes[
        "dataHoraUltimaMedicao"
    ] = pd.to_datetime(

        estacoes[
            "dataHoraUltimaMedicao"
        ],

        errors=
            "coerce",

        utc=
            True,
    )


if (
    "anomalia"
    not in estacoes.columns
):

    estacoes[
        "anomalia"
    ] = ""


# =============================================================================
# COTAS EM METROS
# =============================================================================

for origem, destino in [

    (
        "cotaUltimaMedicao",
        "cotaUltimaMedicao_m",
    ),

    (
        "cotaAlerta",
        "cotaAlerta_m",
    ),

    (
        "cotaInundacao",
        "cotaInundacao_m",
    ),

    (
        "cotaAtencaoInundacao",
        "cotaAtencaoInundacao_m",
    ),

    (
        "cotaAtencaoEstiagem",
        "cotaAtencaoEstiagem_m",
    ),
]:

    if (
        destino not in estacoes.columns
        and
        origem in estacoes.columns
    ):

        estacoes[
            destino
        ] = (
            pd.to_numeric(

                estacoes[
                    origem
                ],

                errors=
                    "coerce",
            )

            /
            100.0
        )


for coluna in [

    "atual_m",
    "maximo_m",
    "media_m",
    "minimo_m",
]:

    if coluna not in historico.columns:

        origem = coluna.replace(
            "_m",
            ""
        )

        if origem in historico.columns:

            historico[
                coluna
            ] = (
                pd.to_numeric(

                    historico[
                        origem
                    ],

                    errors=
                        "coerce",
                )

                /
                100.0
            )


# =============================================================================
# ESTAÇÕES GEO
# =============================================================================

estacoes_geo = estacoes[

    estacoes[
        "latitude"
    ].notna()

    &

    estacoes[
        "longitude"
    ].notna()

].copy()


estacoes_geo[
    "_id"
] = (
    "EST_"
    +
    estacoes_geo[
        "codigo"
    ]
)


estacoes_geo[
    "geometry"
] = [

    Point(
        lon,
        lat,
    )

    for lon, lat
    in zip(

        estacoes_geo[
            "longitude"
        ],

        estacoes_geo[
            "latitude"
        ],
    )
]


estacoes_geo = gpd.GeoDataFrame(

    estacoes_geo,

    geometry=
        "geometry",

    crs=
        CRS_GEO,
)


estacoes_geo = enriquecer_pontos(
    estacoes_geo
)


ESTACOES_PROJ = estacoes_geo.to_crs(
    CRS_METRICO
)


# =============================================================================
# CONTEXTO HISTÓRICO
# =============================================================================

def contexto_historico(
    codigo,
):

    serie = historico[
        historico[
            "codigo"
        ]
        ==
        str(
            codigo
        )
    ].copy()


    if serie.empty:

        return {

            "minimo":
                np.nan,

            "media":
                np.nan,

            "maximo":
                np.nan,

            "delta_1d":
                np.nan,

            "delta_7d":
                np.nan,

            "tendencia":
                "Sem informação",
        }


    serie = serie.sort_values(
        "data"
    )


    clima = serie[
        [
            "data",
            "minimo_m",
            "media_m",
            "maximo_m",
        ]
    ].dropna(

        subset=[
            "minimo_m",
            "media_m",
            "maximo_m",
        ],

        how=
            "all",
    )


    if clima.empty:

        minimo = np.nan
        media = np.nan
        maximo = np.nan

    else:

        ultima_clima = clima.iloc[
            -1
        ]

        minimo = ultima_clima[
            "minimo_m"
        ]

        media = ultima_clima[
            "media_m"
        ]

        maximo = ultima_clima[
            "maximo_m"
        ]


    obs = serie[
        [
            "data",
            "atual_m",
        ]
    ].dropna()


    delta_1d = np.nan
    delta_7d = np.nan
    tendencia = "Sem informação"


    if not obs.empty:

        ultima = obs.iloc[
            -1
        ]


        data_ultima = ultima[
            "data"
        ]


        atual = ultima[
            "atual_m"
        ]


        ant1 = obs[
            obs[
                "data"
            ]
            <=
            (
                data_ultima
                -
                pd.Timedelta(
                    days=1
                )
            )
        ]


        if not ant1.empty:

            delta_1d = (
                atual
                -
                ant1[
                    "atual_m"
                ].iloc[-1]
            )


        ant7 = obs[
            obs[
                "data"
            ]
            <=
            (
                data_ultima
                -
                pd.Timedelta(
                    days=7
                )
            )
        ]


        if not ant7.empty:

            delta_7d = (
                atual
                -
                ant7[
                    "atual_m"
                ].iloc[-1]
            )


        delta_ref = (
            delta_7d
            if pd.notna(
                delta_7d
            )
            else
            delta_1d
        )


        if pd.notna(
            delta_ref
        ):

            if delta_ref > 0.05:

                tendencia = (
                    "Subindo"
                )

            elif delta_ref < -0.05:

                tendencia = (
                    "Descendo"
                )

            else:

                tendencia = (
                    "Estável"
                )


    return {

        "minimo":
            minimo,

        "media":
            media,

        "maximo":
            maximo,

        "delta_1d":
            delta_1d,

        "delta_7d":
            delta_7d,

        "tendencia":
            tendencia,
    }


# =============================================================================
# CONDIÇÃO HIDROLÓGICA
# =============================================================================

def classificar_condicao_hidrologica(
    linha,
):

    codigo_linha = str(
        linha.get(
            "codigo",
            ""
        )
    )

    # ANA: utiliza a cota atual e a referência histórica mensal calculada
    # a partir das médias mensais anuais (2007-2025). A lógica de posição
    # entre mínima/média/máxima é a mesma usada como complemento no CENSIPAM.
    if codigo_linha.startswith(
        "ANA:"
    ):

        minimo = pd.to_numeric(
            linha.get(
                "menor_media_mensal_m",
                np.nan,
            ),
            errors=
                "coerce",
        )

        media = pd.to_numeric(
            linha.get(
                "media_mensal_historica_m",
                np.nan,
            ),
            errors=
                "coerce",
        )

        maximo = pd.to_numeric(
            linha.get(
                "maior_media_mensal_m",
                np.nan,
            ),
            errors=
                "coerce",
        )

        cota = pd.to_numeric(
            linha.get(
                "cotaUltimaMedicao_m",
                np.nan,
            ),
            errors=
                "coerce",
        )

        n_anos = pd.to_numeric(
            linha.get(
                "n_anos_validos",
                np.nan,
            ),
            errors=
                "coerce",
        )

        referencia_suficiente = bool(
            linha.get(
                "referencia_suficiente",
                False,
            )
        )

        nome_mes = texto_valido(
            linha.get(
                "nome_mes",
                "",
            )
        )

        periodo_ref = texto_valido(
            linha.get(
                "periodo_referencia",
                "",
            )
        )

        base = {
            "minimo":
                minimo,

            "media":
                media,

            "maximo":
                maximo,

            "delta_1d":
                np.nan,

            "delta_7d":
                np.nan,

            "tendencia":
                "Sem informação",
        }

        if pd.isna(
            cota
        ):

            return {
                **base,

                "condicao":
                    "SEM CLASSIFICAÇÃO",

                "fonte":
                    "Sem cota atual",

                "descricao":
                    "Não há cota atual disponível para esta estação.",
            }

        if (
            not referencia_suficiente
            or
            pd.isna(minimo)
            or
            pd.isna(media)
            or
            pd.isna(maximo)
        ):

            return {
                **base,

                "condicao":
                    "SEM CLASSIFICAÇÃO",

                "fonte":
                    "Referência histórica mensal insuficiente",

                "descricao":
                    (
                        f"Cota atual: {fmt_numero(cota)} m. "
                        "A estação possui observação atual, mas não há "
                        "referência histórica mensal suficiente para comparação."
                    ),
            }

        condicao = "DENTRO DO ESPERADO"

        # Faixa inferior: mesma lógica relativa usada no complemento CENSIPAM.
        if (
            media > minimo
            and
            cota <= media
        ):

            posicao = (
                (cota - minimo)
                /
                (media - minimo)
            )

            if (
                cota <= minimo
                or
                posicao <= 0.25
            ):

                condicao = "MUITO ABAIXO DO ESPERADO"

            elif posicao <= 0.50:

                condicao = "ABAIXO DO ESPERADO"

            else:

                condicao = "DENTRO DO ESPERADO"

        # Faixa superior: mesma lógica relativa usada no complemento CENSIPAM.
        elif (
            maximo > media
            and
            cota > media
        ):

            posicao = (
                (cota - media)
                /
                (maximo - media)
            )

            if (
                cota >= maximo
                or
                posicao >= 0.75
            ):

                condicao = "MUITO ACIMA DO ESPERADO"

            elif posicao >= 0.50:

                condicao = "ACIMA DO ESPERADO"

            else:

                condicao = "DENTRO DO ESPERADO"

        descricao = (
            f"Cota atual: {fmt_numero(cota)} m. "
            f"Referência de {nome_mes or 'mês corrente'}: "
            f"menor média mensal {fmt_numero(minimo)} m, "
            f"média histórica mensal {fmt_numero(media)} m e "
            f"maior média mensal {fmt_numero(maximo)} m."
        )

        if pd.notna(
            n_anos
        ):

            descricao += (
                f" Base histórica: {int(n_anos)} anos válidos"
            )

            if periodo_ref:

                descricao += (
                    f" ({periodo_ref})"
                )

            descricao += "."

        return {
            **base,

            "condicao":
                condicao,

            "fonte":
                "Referência histórica mensal calculada a partir das séries ANA",

            "descricao":
                descricao,
        }


    contexto = contexto_historico(
        linha[
            "codigo"
        ]
    )


    cota = linha.get(
        "cotaUltimaMedicao_m",
        np.nan,
    )


    if pd.isna(
        cota
    ):

        return {

            **contexto,

            "condicao":
                "SEM CLASSIFICAÇÃO",

            "fonte":
                "Sem cota atual",

            "descricao":
                "Não há cota atual disponível.",
        }


    inundacao = linha.get(
        "cotaInundacao_m",
        np.nan,
    )

    alerta = linha.get(
        "cotaAlerta_m",
        np.nan,
    )

    atencao_inundacao = linha.get(
        "cotaAtencaoInundacao_m",
        np.nan,
    )

    atencao_estiagem = linha.get(
        "cotaAtencaoEstiagem_m",
        np.nan,
    )


    # =========================================================================
    # LIMIARES OFICIAIS
    # =========================================================================

    if (
        pd.notna(
            inundacao
        )
        and
        cota >= inundacao
    ):

        return {

            **contexto,

            "condicao":
                "INUNDAÇÃO",

            "fonte":
                "Limiar oficial SipamHidro",

            "descricao":
                (
                    f"Cota atual ({fmt_numero(cota)} m) igual ou superior "
                    f"à cota oficial de inundação "
                    f"({fmt_numero(inundacao)} m)."
                ),
        }


    if (
        pd.notna(
            alerta
        )
        and
        cota >= alerta
    ):

        return {

            **contexto,

            "condicao":
                "ALERTA — CHEIA",

            "fonte":
                "Limiar oficial SipamHidro",

            "descricao":
                (
                    f"Cota atual ({fmt_numero(cota)} m) igual ou superior "
                    f"à cota oficial de alerta ({fmt_numero(alerta)} m)."
                ),
        }


    if (
        pd.notna(
            atencao_inundacao
        )
        and
        cota >= atencao_inundacao
    ):

        return {

            **contexto,

            "condicao":
                "ATENÇÃO — CHEIA",

            "fonte":
                "Limiar oficial SipamHidro",

            "descricao":
                (
                    f"Cota atual ({fmt_numero(cota)} m) igual ou superior "
                    f"à atenção para inundação "
                    f"({fmt_numero(atencao_inundacao)} m)."
                ),
        }


    if (
        pd.notna(
            atencao_estiagem
        )
        and
        cota <= atencao_estiagem
    ):

        return {

            **contexto,

            "condicao":
                "ATENÇÃO — ESTIAGEM",

            "fonte":
                "Limiar oficial SipamHidro",

            "descricao":
                (
                    f"Cota atual ({fmt_numero(cota)} m) igual ou inferior "
                    f"à atenção para estiagem "
                    f"({fmt_numero(atencao_estiagem)} m)."
                ),
        }


    # =========================================================================
    # CLIMATOLOGIA
    # =========================================================================

    minimo = contexto[
        "minimo"
    ]

    media = contexto[
        "media"
    ]

    maximo = contexto[
        "maximo"
    ]

    tendencia = contexto[
        "tendencia"
    ]

    anomalia = str(
        linha.get(
            "anomalia",
            ""
        )
    ).upper()


    # BAIXA
    if (
        pd.notna(
            minimo
        )
        and
        pd.notna(
            media
        )
        and
        media > minimo
        and
        cota <= media
    ):

        posicao = (
            (
                cota
                -
                minimo
            )
            /
            (
                media
                -
                minimo
            )
        )


        if (
            cota <= minimo
            or
            posicao <= 0.25
        ):

            condicao = (
                "MUITO ABAIXO DO ESPERADO"
            )


        elif posicao <= 0.50:

            condicao = (
                "ABAIXO DO ESPERADO"
            )


        else:

            condicao = (
                "DENTRO DO ESPERADO"
            )


        if (
            tendencia
            ==
            "Descendo"
            and
            "NEGATIVA"
            in anomalia
            and
            posicao <= 0.50
        ):

            condicao = (
                "MUITO ABAIXO DO ESPERADO"
            )


        return {

            **contexto,

            "condicao":
                condicao,

            "fonte":
                "Condição climatológica complementar",

            "descricao":
                (
                    f"Cota atual: {fmt_numero(cota)} m. "
                    f"Mínima histórica: {fmt_numero(minimo)} m. "
                    f"Média histórica: {fmt_numero(media)} m. "
                    f"Tendência recente: {tendencia}."
                ),
        }


    # ALTA
    if (
        pd.notna(
            media
        )
        and
        pd.notna(
            maximo
        )
        and
        maximo > media
        and
        cota > media
    ):

        posicao = (
            (
                cota
                -
                media
            )
            /
            (
                maximo
                -
                media
            )
        )


        if (
            cota >= maximo
            or
            posicao >= 0.75
        ):

            condicao = (
                "MUITO ACIMA DO ESPERADO"
            )


        elif posicao >= 0.50:

            condicao = (
                "ACIMA DO ESPERADO"
            )


        else:

            condicao = (
                "DENTRO DO ESPERADO"
            )


        if (
            tendencia
            ==
            "Subindo"
            and
            "POSITIVA"
            in anomalia
            and
            posicao >= 0.50
        ):

            condicao = (
                "MUITO ACIMA DO ESPERADO"
            )


        return {

            **contexto,

            "condicao":
                condicao,

            "fonte":
                "Condição climatológica complementar",

            "descricao":
                (
                    f"Cota atual: {fmt_numero(cota)} m. "
                    f"Média histórica: {fmt_numero(media)} m. "
                    f"Máxima histórica: {fmt_numero(maximo)} m. "
                    f"Tendência recente: {tendencia}."
                ),
        }


    return {

        **contexto,

        "condicao":
            "DENTRO DO ESPERADO",

        "fonte":
            "Monitoramento hidrológico",

        "descricao":
            (
                "Nenhum limiar oficial ou sinal climatológico "
                "relevante foi identificado."
            ),
    }


# =============================================================================
# PRÉ-CÁLCULO
# =============================================================================

print(
    "Calculando condição hidrológica..."
)


resultados = []


for _, linha in estacoes_geo.iterrows():

    resultados.append(

        classificar_condicao_hidrologica(
            linha
        )
    )


estacoes_geo[
    "condicao_hidrologica"
] = [

    x[
        "condicao"
    ]

    for x
    in resultados
]


estacoes_geo[
    "tendencia"
] = [

    x[
        "tendencia"
    ]

    for x
    in resultados
]


estacoes_geo[
    "delta_1d"
] = [

    x[
        "delta_1d"
    ]

    for x
    in resultados
]


estacoes_geo[
    "delta_7d"
] = [

    x[
        "delta_7d"
    ]

    for x
    in resultados
]


ESTACOES_PROJ = estacoes_geo.to_crs(
    CRS_METRICO
)


# =============================================================================
# CARREGAR BHO SOMENTE NO RAIO ESCOLHIDO
# =============================================================================

@lru_cache(
    maxsize=20
)

def carregar_bho_local(
    codigo_estacao,
    raio_km,
):

    # No Render, o BHO250 fica fora do repositório por causa do tamanho.
    # Retornamos uma camada vazia para manter o restante do painel operacional.
    if not BHO_DISPONIVEL:
        return gpd.GeoDataFrame(geometry=[], crs=CRS_GEO)

    meta = estacoes_geo[
        estacoes_geo[
            "codigo"
        ]
        ==
        str(
            codigo_estacao
        )
    ]


    if meta.empty:

        return gpd.GeoDataFrame(
            geometry=[],
            crs=CRS_GEO,
        )


    linha = meta.iloc[
        0
    ]


    lat = float(
        linha[
            "latitude"
        ]
    )

    lon = float(
        linha[
            "longitude"
        ]
    )


    margem_lat = (
        float(
            raio_km
        )
        /
        111.0
        *
        1.10
    )


    cos_lat = max(

        math.cos(
            math.radians(
                lat
            )
        ),

        0.20,
    )


    margem_lon = (
        float(
            raio_km
        )
        /
        (
            111.0
            *
            cos_lat
        )
        *
        1.10
    )


    bbox = (

        lon
        -
        margem_lon,

        lat
        -
        margem_lat,

        lon
        +
        margem_lon,

        lat
        +
        margem_lat,
    )


    print(
        "\n"
        +
        "=" * 80
    )

    print(
        f"BHO250 | Estação {codigo_estacao} | Raio {raio_km} km"
    )

    print(
        "=" * 80
    )


    bho = gpd.read_file(

        ARQ_BHO,

        layer=
            CAMADA_BHO,

        bbox=
            bbox,
    )


    if bho.empty:

        return bho


    if bho.crs is None:

        bho = bho.set_crs(
            CRS_BHO
        )


    bho = bho.to_crs(
        CRS_GEO
    )


    # =========================================================================
    # RECORTE REAL PELO CÍRCULO
    # =========================================================================

    est = ESTACOES_PROJ[
        ESTACOES_PROJ[
            "codigo"
        ]
        ==
        str(
            codigo_estacao
        )
    ]


    ponto = est.geometry.iloc[
        0
    ]


    buffer = ponto.buffer(
        float(
            raio_km
        )
        *
        1000
    )


    bho_proj = bho.to_crs(
        CRS_METRICO
    )


    bho_proj = bho_proj[
        bho_proj
        .geometry
        .intersects(
            buffer
        )
    ].copy()


    print(
        f"Trechos carregados no raio: "
        f"{len(bho_proj):,}"
    )


    return bho_proj.to_crs(
        CRS_GEO
    )


# =============================================================================
# REDE CONECTADA DENTRO DO MESMO RAIO
# =============================================================================

def identificar_rede_conectada(
    codigo_estacao,
    raio_km,
):

    bho = carregar_bho_local(

        str(
            codigo_estacao
        ),

        int(
            raio_km
        ),
    ).copy()


    if bho.empty:

        return (
            bho,
            bho,
            None,
        )


    bho_proj = bho.to_crs(
        CRS_METRICO
    )


    est = ESTACOES_PROJ[
        ESTACOES_PROJ[
            "codigo"
        ]
        ==
        str(
            codigo_estacao
        )
    ]


    ponto = est.geometry.iloc[
        0
    ]


    # =========================================================================
    # TRECHO MAIS PRÓXIMO DA ESTAÇÃO
    # =========================================================================

    distancias = (
        bho_proj
        .geometry
        .distance(
            ponto
        )
    )


    idx_inicio = distancias.idxmin()


    trecho_inicio = bho.loc[
        idx_inicio
    ]


    no_origem_inicio = chave_no(
        trecho_inicio.get(
            "noorigem"
        )
    )


    no_destino_inicio = chave_no(
        trecho_inicio.get(
            "nodestino"
        )
    )


    # =========================================================================
    # MAPAS TOPOLÓGICOS
    # =========================================================================

    origem_map = {}

    destino_map = {}


    for idx, row in bho.iterrows():

        noorigem = chave_no(
            row.get(
                "noorigem"
            )
        )

        nodestino = chave_no(
            row.get(
                "nodestino"
            )
        )


        if noorigem is not None:

            origem_map.setdefault(
                noorigem,
                [],
            ).append(
                idx
            )


        if nodestino is not None:

            destino_map.setdefault(
                nodestino,
                [],
            ).append(
                idx
            )


    # =========================================================================
    # JUSANTE
    # =========================================================================

    jusante = set()

    fila = []

    if no_destino_inicio is not None:

        fila.append(
            no_destino_inicio
        )


    nos_visitados = set()


    while fila:

        no = fila.pop(
            0
        )


        if no in nos_visitados:

            continue


        nos_visitados.add(
            no
        )


        for idx in origem_map.get(
            no,
            [],
        ):

            if idx == idx_inicio:

                continue


            if idx in jusante:

                continue


            jusante.add(
                idx
            )


            prox = chave_no(
                bho.loc[
                    idx
                ].get(
                    "nodestino"
                )
            )


            if prox is not None:

                fila.append(
                    prox
                )


            if (
                len(
                    jusante
                )
                >=
                MAX_TRECHOS_CONECTADOS
            ):

                break


        if (
            len(
                jusante
            )
            >=
            MAX_TRECHOS_CONECTADOS
        ):

            break


    # =========================================================================
    # MONTANTE
    # =========================================================================

    montante = set()

    fila = []


    if no_origem_inicio is not None:

        fila.append(
            no_origem_inicio
        )


    nos_visitados = set()


    while fila:

        no = fila.pop(
            0
        )


        if no in nos_visitados:

            continue


        nos_visitados.add(
            no
        )


        for idx in destino_map.get(
            no,
            [],
        ):

            if idx == idx_inicio:

                continue


            if idx in montante:

                continue


            montante.add(
                idx
            )


            ant = chave_no(
                bho.loc[
                    idx
                ].get(
                    "noorigem"
                )
            )


            if ant is not None:

                fila.append(
                    ant
                )


            if (
                len(
                    montante
                )
                >=
                MAX_TRECHOS_CONECTADOS
            ):

                break


        if (
            len(
                montante
            )
            >=
            MAX_TRECHOS_CONECTADOS
        ):

            break


    # =========================================================================
    # REDE FINAL
    # =========================================================================

    indices_rede = (

        {
            idx_inicio
        }

        |

        montante

        |

        jusante
    )


    rede = bho.loc[
        list(
            indices_rede
        )
    ].copy()


    rede[
        "relacao_rede"
    ] = "Rede conectada"


    if len(
        montante
    ) > 0:

        rede.loc[
            list(
                montante
            ),

            "relacao_rede",
        ] = "Montante"


    if len(
        jusante
    ) > 0:

        rede.loc[
            list(
                jusante
            ),

            "relacao_rede",
        ] = "Jusante"


    rede.loc[
        idx_inicio,

        "relacao_rede",
    ] = "Trecho da estação"


    info = {

        "nome_rio":
            obter_nome_rio(
                trecho_inicio
            ),

        "cotrecho":
            trecho_inicio.get(
                "cotrecho",
                "—",
            ),

        "nome_bacia":
            obter_nome_bacia(
                trecho_inicio
            ),

        "distancia_m":
            float(
                distancias.loc[
                    idx_inicio
                ]
            ),

        "n_total":
            len(
                rede
            ),

        "n_montante":
            len(
                montante
            ),

        "n_jusante":
            len(
                jusante
            ),
    }


    return (
        bho,
        rede,
        info,
    )


# =============================================================================
# UNIDADES NO RAIO
# =============================================================================

def unidades_no_entorno(
    codigo,
    raio_km,
):

    est = ESTACOES_PROJ[
        ESTACOES_PROJ[
            "codigo"
        ]
        ==
        str(
            codigo
        )
    ]


    if est.empty:

        return gpd.GeoDataFrame()


    ponto = est.geometry.iloc[
        0
    ]


    buffer = ponto.buffer(
        float(
            raio_km
        )
        *
        1000
    )


    indices = SINDEX_SAUDE.query(

        buffer,

        predicate=
            "intersects",
    )


    if len(
        indices
    ) == 0:

        return gpd.GeoDataFrame()


    candidatos = (
        SAUDE_PROJ
        .iloc[
            indices
        ]
        .copy()
    )


    candidatos[
        "dist_km"
    ] = (

        candidatos
        .geometry
        .distance(
            ponto
        )

        /
        1000
    )


    candidatos = candidatos[
        candidatos[
            "dist_km"
        ]
        <=
        float(
            raio_km
        )
    ].copy()


    candidatos = candidatos.sort_values(
        "dist_km"
    )


    return candidatos


# =============================================================================
# TABELA DAS UNIDADES
# =============================================================================

def tabela_unidades(
    codigo,
    raio_km,
):

    unidades = unidades_no_entorno(
        codigo,
        raio_km,
    )


    if unidades.empty:

        return pd.DataFrame(

            columns=[

                "Tipo",
                "Unidade",
                "CNES",
                "Município",
                "UF",
                "Polo Base",
                "DSEI",
                "Distância da estação (km)",
            ]
        )


    return pd.DataFrame({

        "Tipo":
            unidades[
                "tipo"
            ].values,

        "Unidade":
            unidades[
                "nome"
            ].values,

        "CNES":
            unidades[
                "cnes"
            ].values,

        "Município":
            unidades[
                "municipio"
            ].values,

        "UF":
            unidades[
                "uf"
            ].values,

        "Polo Base":
            unidades[
                "polo_base"
            ].values,

        "DSEI":
            unidades[
                "dsei"
            ].values,

        "Distância da estação (km)":
            unidades[
                "dist_km"
            ]
            .round(
                2
            )
            .values,
    })


# =============================================================================
# POLOS NO RAIO
# =============================================================================

def polos_no_entorno(
    codigo,
    raio_km,
):

    est = ESTACOES_PROJ[
        ESTACOES_PROJ[
            "codigo"
        ]
        ==
        str(
            codigo
        )
    ]


    if est.empty:

        return pd.DataFrame()


    ponto = est.geometry.iloc[
        0
    ]


    distancias = (
        POLOS_PROJ
        .geometry
        .distance(
            ponto
        )
    )


    limite = (
        float(
            raio_km
        )
        *
        1000
    )


    subset = POLOS_PROJ[
        distancias
        <=
        limite
    ].copy()


    if subset.empty:

        return pd.DataFrame()


    subset[
        "dist_km"
    ] = (

        distancias.loc[
            subset.index
        ]

        /
        1000
    )


    subset = subset.sort_values(
        "dist_km"
    )


    return pd.DataFrame({

        "Polo Base":
            subset[
                "polo_nome"
            ].values,

        "DSEI":
            subset[
                "dsei_nome"
            ].values,

        "UF":
            subset[
                "polo_uf"
            ].values,

        "Distância da estação (km)":
            subset[
                "dist_km"
            ]
            .round(
                2
            )
            .values,
    })


# =============================================================================
# LIMIARES OFICIAIS
# =============================================================================

def tabela_limites_oficiais(
    linha,
):

    atual = linha.get(
        "cotaUltimaMedicao_m",
        np.nan,
    )


    regras = [

        (
            "Atenção à estiagem",
            "cotaAtencaoEstiagem_m",
            "<=",
        ),

        (
            "Atenção à inundação",
            "cotaAtencaoInundacao_m",
            ">=",
        ),

        (
            "Alerta",
            "cotaAlerta_m",
            ">=",
        ),

        (
            "Inundação",
            "cotaInundacao_m",
            ">=",
        ),
    ]


    dados = []


    for nome, campo, operador in regras:

        limite = linha.get(
            campo,
            np.nan,
        )


        if pd.isna(
            limite
        ):

            dados.append({

                "Limiar":
                    nome,

                "Valor":
                    "Sem limiar informado",

                "Situação":
                    "Sem limiar informado",
            })

            continue


        if pd.isna(
            atual
        ):

            situacao = (
                "Sem cota atual"
            )


        elif operador == ">=":

            situacao = (
                "ACIONADO"
                if atual >= limite
                else
                "Não acionado"
            )


        else:

            situacao = (
                "ACIONADO"
                if atual <= limite
                else
                "Não acionado"
            )


        dados.append({

            "Limiar":
                nome,

            "Valor":
                (
                    f"{fmt_numero(limite)} m"
                ),

            "Situação":
                situacao,
        })


    return dados



# =============================================================================
# ANA/SNIRH - ESTAÇÕES COM COTA ATUAL
# =============================================================================

print("Lendo estações ANA/SNIRH com cota atual...")

if ARQ_ANA_ATUAL.exists():

    ANA_RAW = pd.read_parquet(
        ARQ_ANA_ATUAL
    )

else:

    print(
        f"ATENÇÃO: arquivo ANA não encontrado: {ARQ_ANA_ATUAL}"
    )

    ANA_RAW = pd.DataFrame()


def _col_ana(
    df,
    candidatos,
):

    mapa = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for candidato in candidatos:

        chave = (
            str(candidato)
            .lower()
            .strip()
        )

        if chave in mapa:

            return mapa[
                chave
            ]

    return None


if not ANA_RAW.empty:

    C_ANA_COD = _col_ana(
        ANA_RAW,
        [
            "codigo",
            "codestacao",
            "codigo_estacao",
        ],
    )

    C_ANA_NOME = _col_ana(
        ANA_RAW,
        [
            "nome",
            "nomeestacao",
            "nome_estacao",
        ],
    )

    C_ANA_RIO = _col_ana(
        ANA_RAW,
        [
            "rio",
            "nomerio",
            "nome_rio",
        ],
    )

    C_ANA_MUN = _col_ana(
        ANA_RAW,
        [
            "municipio",
            "municipio_uf",
        ],
    )

    C_ANA_UF = _col_ana(
        ANA_RAW,
        [
            "uf",
            "estado",
        ],
    )

    C_ANA_LAT = _col_ana(
        ANA_RAW,
        [
            "latitude",
            "lat",
        ],
    )

    C_ANA_LON = _col_ana(
        ANA_RAW,
        [
            "longitude",
            "lon",
            "lng",
        ],
    )

    C_ANA_COTA_M = _col_ana(
        ANA_RAW,
        [
            "ultimo_nivel_m",
            "cota_atual_m",
            "nivel_m",
        ],
    )

    C_ANA_COTA_CM = _col_ana(
        ANA_RAW,
        [
            "ultimo_nivel_cm",
            "nivel",
            "nivel_cm",
        ],
    )

    C_ANA_DATA = _col_ana(
        ANA_RAW,
        [
            "ultimo_nivel_data",
            "data_cota_atual",
            "datahora",
            "data_hora",
        ],
    )

    ANA = pd.DataFrame(
        index=
            ANA_RAW.index
    )

    ANA[
        "codigo"
    ] = (
        ANA_RAW[
            C_ANA_COD
        ]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()

        if C_ANA_COD

        else
        ANA_RAW.index.astype(str)
    )

    ANA[
        "nome"
    ] = (
        ANA_RAW[
            C_ANA_NOME
        ]
        .fillna("")
        .astype(str)

        if C_ANA_NOME

        else
        "Estação ANA"
    )

    ANA[
        "rio"
    ] = (
        ANA_RAW[
            C_ANA_RIO
        ]
        .fillna("")
        .astype(str)

        if C_ANA_RIO

        else
        ""
    )

    ANA[
        "municipio"
    ] = (
        ANA_RAW[
            C_ANA_MUN
        ]
        .fillna("")
        .astype(str)

        if C_ANA_MUN

        else
        ""
    )

    ANA[
        "uf"
    ] = (
        ANA_RAW[
            C_ANA_UF
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()

        if C_ANA_UF

        else
        ""
    )

    ANA[
        "latitude"
    ] = (
        pd.to_numeric(
            ANA_RAW[
                C_ANA_LAT
            ],
            errors=
                "coerce",
        )

        if C_ANA_LAT

        else
        np.nan
    )

    ANA[
        "longitude"
    ] = (
        pd.to_numeric(
            ANA_RAW[
                C_ANA_LON
            ],
            errors=
                "coerce",
        )

        if C_ANA_LON

        else
        np.nan
    )

    if C_ANA_COTA_M:

        ANA[
            "cota_m"
        ] = pd.to_numeric(
            ANA_RAW[
                C_ANA_COTA_M
            ],
            errors=
                "coerce",
        )

    elif C_ANA_COTA_CM:

        ANA[
            "cota_m"
        ] = (
            pd.to_numeric(
                ANA_RAW[
                    C_ANA_COTA_CM
                ],
                errors=
                    "coerce",
            )
            /
            100.0
        )

    else:

        ANA[
            "cota_m"
        ] = np.nan

    ANA[
        "data"
    ] = (
        pd.to_datetime(
            ANA_RAW[
                C_ANA_DATA
            ],
            errors=
                "coerce",
        )

        if C_ANA_DATA

        else
        pd.NaT
    )

    # Mantém SOMENTE estações ANA que realmente possuem informação útil:
    # cota atual + coordenadas.
    ANA = ANA[
        ANA[
            "cota_m"
        ].notna()
        &
        ANA[
            "latitude"
        ].notna()
        &
        ANA[
            "longitude"
        ].notna()
    ].copy()

    ANA = (
        ANA
        .sort_values(
            [
                "uf",
                "nome",
            ]
        )
        .drop_duplicates(
            "codigo",
            keep=
                "last",
        )
        .reset_index(
            drop=True
        )
    )

else:

    ANA = pd.DataFrame(
        columns=[
            "codigo",
            "nome",
            "rio",
            "municipio",
            "uf",
            "latitude",
            "longitude",
            "cota_m",
            "data",
        ]
    )


print(
    f"Estações ANA com cota atual e coordenadas: {len(ANA):,}"
)


# =============================================================================
# =============================================================================
# COMPLEMENTO ANA - REFERÊNCIA HISTÓRICA MENSAL
# =============================================================================

if (
    ARQ_ANA_REFERENCIA_MENSAL.exists()
    and
    not ANA.empty
):

    _ana_ref = pd.read_parquet(
        ARQ_ANA_REFERENCIA_MENSAL
    ).copy()

    _ana_ref[
        "codigo"
    ] = (
        _ana_ref[
            "codigo"
        ]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )

    # O mês de comparação é o mês da observação atual da estação.
    ANA[
        "mes_referencia_atual"
    ] = pd.to_datetime(
        ANA[
            "data"
        ],
        errors=
            "coerce",
    ).dt.month

    _cols_ref = [
        c
        for c
        in [
            "codigo",
            "mes",
            "nome_mes",
            "menor_media_mensal_m",
            "media_mensal_historica_m",
            "mediana_mensal_historica_m",
            "maior_media_mensal_m",
            "desvio_padrao_medias_mensais_m",
            "p25_medias_mensais_m",
            "p75_medias_mensais_m",
            "n_anos_validos",
            "primeiro_ano_valido",
            "ultimo_ano_valido",
            "cobertura_media",
            "referencia_suficiente",
            "periodo_referencia",
        ]
        if c
        in _ana_ref.columns
    ]

    _ana_ref_atual = (
        _ana_ref[
            _cols_ref
        ]
        .drop_duplicates(
            [
                "codigo",
                "mes",
            ],
            keep=
                "last",
        )
    )

    ANA = ANA.merge(
        _ana_ref_atual,
        left_on=[
            "codigo",
            "mes_referencia_atual",
        ],
        right_on=[
            "codigo",
            "mes",
        ],
        how=
            "left",
    )

else:

    ANA[
        "mes_referencia_atual"
    ] = pd.to_datetime(
        ANA[
            "data"
        ],
        errors=
            "coerce",
    ).dt.month


for _c in [
    "menor_media_mensal_m",
    "media_mensal_historica_m",
    "mediana_mensal_historica_m",
    "maior_media_mensal_m",
    "desvio_padrao_medias_mensais_m",
    "p25_medias_mensais_m",
    "p75_medias_mensais_m",
    "n_anos_validos",
    "primeiro_ano_valido",
    "ultimo_ano_valido",
    "cobertura_media",
]:

    if _c not in ANA.columns:

        ANA[
            _c
        ] = np.nan


if "referencia_suficiente" not in ANA.columns:

    ANA[
        "referencia_suficiente"
    ] = False

else:

    ANA[
        "referencia_suficiente"
    ] = (
        ANA[
            "referencia_suficiente"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )


if "nome_mes" not in ANA.columns:

    ANA[
        "nome_mes"
    ] = ""


if "periodo_referencia" not in ANA.columns:

    ANA[
        "periodo_referencia"
    ] = ""


# A condição ANA é calculada com a mesma lógica relativa usada no complemento
# climatológico CENSIPAM, porém com a referência histórica mensal ANA.
def _classificar_ana_mensal(
    row,
):

    cota = pd.to_numeric(
        row.get(
            "cota_m",
            np.nan,
        ),
        errors=
            "coerce",
    )

    minimo = pd.to_numeric(
        row.get(
            "menor_media_mensal_m",
            np.nan,
        ),
        errors=
            "coerce",
    )

    media = pd.to_numeric(
        row.get(
            "media_mensal_historica_m",
            np.nan,
        ),
        errors=
            "coerce",
    )

    maximo = pd.to_numeric(
        row.get(
            "maior_media_mensal_m",
            np.nan,
        ),
        errors=
            "coerce",
    )

    if (
        pd.isna(cota)
        or
        not bool(
            row.get(
                "referencia_suficiente",
                False,
            )
        )
        or
        pd.isna(minimo)
        or
        pd.isna(media)
        or
        pd.isna(maximo)
    ):

        return "SEM CLASSIFICAÇÃO"

    if (
        media > minimo
        and
        cota <= media
    ):

        posicao = (
            (cota - minimo)
            /
            (media - minimo)
        )

        if (
            cota <= minimo
            or
            posicao <= 0.25
        ):

            return "MUITO ABAIXO DO ESPERADO"

        if posicao <= 0.50:

            return "ABAIXO DO ESPERADO"

        return "DENTRO DO ESPERADO"

    if (
        maximo > media
        and
        cota > media
    ):

        posicao = (
            (cota - media)
            /
            (maximo - media)
        )

        if (
            cota >= maximo
            or
            posicao >= 0.75
        ):

            return "MUITO ACIMA DO ESPERADO"

        if posicao >= 0.50:

            return "ACIMA DO ESPERADO"

        return "DENTRO DO ESPERADO"

    return "DENTRO DO ESPERADO"


ANA[
    "condicao_painel"
] = ANA.apply(
    _classificar_ana_mensal,
    axis=
        1,
)


# Mantém no mapa/dropdown somente estações ANA cuja observação atual
# possui referência mensal suficiente para interpretação.
_n_ana_antes_filtro = len(
    ANA
)

ANA = ANA[
    ANA[
        "condicao_painel"
    ]
    !=
    "SEM CLASSIFICAÇÃO"
].copy()

print(
    "ANA removidas por ausência de referência mensal suficiente: "
    f"{_n_ana_antes_filtro - len(ANA):,}"
)

print(
    f"ANA aptas para a rede unificada: {len(ANA):,}"
)


# Preserva a rede CENSIPAM separadamente apenas para controles internos.
CENSIPAM_GEO = estacoes_geo.copy()

CENSIPAM_GEO[
    "fonte_dado"
] = "CENSIPAM"

CENSIPAM_GEO[
    "codigo_exibicao"
] = CENSIPAM_GEO[
    "codigo"
].astype(str)


# Converte ANA para o mesmo modelo espacial usado pelas funções de entorno.
if not ANA.empty:

    _ana_geo = ANA.copy()

    _ana_geo[
        "codigo_original"
    ] = _ana_geo[
        "codigo"
    ].astype(str)

    _ana_geo[
        "codigo"
    ] = (
        "ANA:"
        +
        _ana_geo[
            "codigo_original"
        ]
    )

    _ana_geo[
        "codigo_exibicao"
    ] = _ana_geo[
        "codigo_original"
    ]

    _ana_geo[
        "cotaUltimaMedicao_m"
    ] = _ana_geo[
        "cota_m"
    ]

    _ana_geo[
        "dataHoraUltimaMedicao"
    ] = _ana_geo[
        "data"
    ]

    _ana_geo[
        "condicao_hidrologica"
    ] = _ana_geo[
        "condicao_painel"
    ]

    _ana_geo[
        "tendencia"
    ] = "Sem informação"

    _ana_geo[
        "delta_1d"
    ] = np.nan

    _ana_geo[
        "delta_7d"
    ] = np.nan

    _ana_geo[
        "anomalia"
    ] = ""

    _ana_geo[
        "fonte_dado"
    ] = "ANA"

    for _limiar in [
        "cotaInundacao_m",
        "cotaAlerta_m",
        "cotaAtencaoInundacao_m",
        "cotaAtencaoEstiagem_m",
    ]:

        _ana_geo[
            _limiar
        ] = np.nan

    _ana_geo[
        "geometry"
    ] = [

        Point(
            lon,
            lat,
        )

        for lon, lat
        in zip(
            _ana_geo[
                "longitude"
            ],
            _ana_geo[
                "latitude"
            ],
        )
    ]

    _ana_geo = gpd.GeoDataFrame(
        _ana_geo,
        geometry=
            "geometry",
        crs=
            CRS_GEO,
    )

    # Enriquece ANA com município/Polo Base pelo mesmo procedimento espacial.
    _ana_geo = enriquecer_pontos(
        _ana_geo
    )

    _ana_geo[
        "mun_nome"
    ] = np.where(
        _ana_geo[
            "mun_nome"
        ].fillna("").astype(str).str.strip()
        !=
        "",
        _ana_geo[
            "mun_nome"
        ],
        _ana_geo[
            "municipio"
        ].fillna(""),
    )

    _ana_geo[
        "mun_uf"
    ] = np.where(
        _ana_geo[
            "mun_uf"
        ].fillna("").astype(str).str.strip()
        !=
        "",
        _ana_geo[
            "mun_uf"
        ],
        _ana_geo[
            "uf"
        ].fillna(""),
    )

    if "polo_nome" not in _ana_geo.columns:
        _ana_geo["polo_nome"] = ""
    else:
        _ana_geo["polo_nome"] = _ana_geo["polo_nome"].fillna("")

    if "dsei_nome" not in _ana_geo.columns:
        _ana_geo["dsei_nome"] = ""
    else:
        _ana_geo["dsei_nome"] = _ana_geo["dsei_nome"].fillna("")


    # =====================================================================
    # REMOÇÃO DE DUPLICATAS ANA x CENSIPAM
    # PRIORIDADE: CENSIPAM
    # =====================================================================

    _n_ana_pre_dedup = len(
        _ana_geo
    )

    # 1) Correspondência direta pelo código da estação.
    _codigos_censipam = set(
        CENSIPAM_GEO[
            "codigo_exibicao"
        ]
        .fillna("")
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )

    _duplicada_codigo = (
        _ana_geo[
            "codigo_exibicao"
        ]
        .fillna("")
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
        .isin(
            _codigos_censipam
        )
    )


    # 2) Verificação espacial conservadora para casos em que o código difere.
    # Considera possível duplicata somente quando:
    # - a estação ANA está a até 100 m de uma CENSIPAM; E
    # - está no mesmo município; E
    # - o nome da estação OU o nome do rio coincide após normalização.
    def _norm_txt_dup(
        valor,
    ):

        if pd.isna(
            valor
        ):

            return ""

        valor = (
            str(
                valor
            )
            .upper()
            .strip()
        )

        trocas = {
            "Á": "A",
            "À": "A",
            "Â": "A",
            "Ã": "A",
            "É": "E",
            "Ê": "E",
            "Í": "I",
            "Ó": "O",
            "Ô": "O",
            "Õ": "O",
            "Ú": "U",
            "Ç": "C",
        }

        for a, b in trocas.items():

            valor = valor.replace(
                a,
                b,
            )

        return " ".join(
            valor.split()
        )


    _ana_metric = _ana_geo.to_crs(
        CRS_METRICO
    )

    _cens_metric = CENSIPAM_GEO.to_crs(
        CRS_METRICO
    )

    _duplicada_espacial = pd.Series(
        False,
        index=
            _ana_geo.index,
        dtype=
            bool,
    )

    if (
        not _ana_metric.empty
        and
        not _cens_metric.empty
    ):

        _idx_cens = _cens_metric.sindex

        for _idx_ana, _row_ana in _ana_metric.iterrows():

            if bool(
                _duplicada_codigo.loc[
                    _idx_ana
                ]
            ):

                continue

            _geom = _row_ana.geometry

            _candidatos = list(
                _idx_cens.query(
                    _geom.buffer(
                        100.0
                    ),
                    predicate=
                        "intersects",
                )
            )

            if not _candidatos:

                continue

            _mun_ana = _norm_txt_dup(
                _ana_geo.loc[
                    _idx_ana
                ].get(
                    "mun_nome",
                    "",
                )
            )

            _nome_ana = _norm_txt_dup(
                _ana_geo.loc[
                    _idx_ana
                ].get(
                    "nome",
                    "",
                )
            )

            _rio_ana = _norm_txt_dup(
                _ana_geo.loc[
                    _idx_ana
                ].get(
                    "rio",
                    "",
                )
            )

            for _pos in _candidatos:

                _row_cens = _cens_metric.iloc[
                    _pos
                ]

                _dist = _geom.distance(
                    _row_cens.geometry
                )

                if _dist > 100.0:

                    continue

                _idx_real_cens = _cens_metric.index[
                    _pos
                ]

                _mun_cens = _norm_txt_dup(
                    CENSIPAM_GEO.loc[
                        _idx_real_cens
                    ].get(
                        "mun_nome",
                        "",
                    )
                )

                if (
                    _mun_ana
                    and
                    _mun_cens
                    and
                    _mun_ana
                    !=
                    _mun_cens
                ):

                    continue

                _nome_cens = _norm_txt_dup(
                    CENSIPAM_GEO.loc[
                        _idx_real_cens
                    ].get(
                        "nome",
                        "",
                    )
                )

                _rio_cens = _norm_txt_dup(
                    CENSIPAM_GEO.loc[
                        _idx_real_cens
                    ].get(
                        "rio",
                        "",
                    )
                )

                _mesmo_nome = (
                    _nome_ana
                    and
                    _nome_cens
                    and
                    _nome_ana
                    ==
                    _nome_cens
                )

                _mesmo_rio = (
                    _rio_ana
                    and
                    _rio_cens
                    and
                    _rio_ana
                    ==
                    _rio_cens
                )

                if (
                    _mesmo_nome
                    or
                    _mesmo_rio
                ):

                    _duplicada_espacial.loc[
                        _idx_ana
                    ] = True

                    break


    _duplicada = (
        _duplicada_codigo
        |
        _duplicada_espacial
    )

    _n_dup_codigo = int(
        _duplicada_codigo.sum()
    )

    _n_dup_espacial = int(
        (
            _duplicada_espacial
            &
            ~_duplicada_codigo
        ).sum()
    )

    _ana_geo = _ana_geo[
        ~_duplicada
    ].copy()

    print(
        "Duplicatas ANA x CENSIPAM removidas: "
        f"{_n_ana_pre_dedup - len(_ana_geo):,} "
        f"(código: {_n_dup_codigo:,}; "
        f"espacial/nome-rio: {_n_dup_espacial:,})"
    )

    print(
        f"Estações ANA mantidas após prioridade CENSIPAM: {len(_ana_geo):,}"
    )


    # Alinha colunas sem eliminar nenhum dado específico de cada rede.
    _todas_colunas = sorted(
        set(
            CENSIPAM_GEO.columns
        )
        |
        set(
            _ana_geo.columns
        )
    )

    for _c in _todas_colunas:

        if _c not in CENSIPAM_GEO.columns:
            CENSIPAM_GEO[
                _c
            ] = np.nan

        if _c not in _ana_geo.columns:
            _ana_geo[
                _c
            ] = np.nan

    estacoes_geo = gpd.GeoDataFrame(
        pd.concat(
            [
                CENSIPAM_GEO[
                    _todas_colunas
                ],
                _ana_geo[
                    _todas_colunas
                ],
            ],
            ignore_index=
                True,
        ),
        geometry=
            "geometry",
        crs=
            CRS_GEO,
    )

else:

    estacoes_geo = CENSIPAM_GEO.copy()


# As funções de raio/saúde/rede usam este objeto.
ESTACOES_PROJ = estacoes_geo.to_crs(
    CRS_METRICO
)

print(
    f"Rede unificada com cota disponível: {len(estacoes_geo):,} estações"
)


def opcoes_ana_por_uf(
    uf=None,
):

    temp = ANA.copy()

    if (
        uf
        and
        uf != "TODAS"
    ):

        temp = temp[
            temp[
                "uf"
            ]
            ==
            uf
        ]

    return [
        {
            "label":
                (
                    f"{linha['nome']} "
                    f"({linha['codigo']})"
                ),
            "value":
                linha[
                    "codigo"
                ],
        }

        for _, linha
        in temp.iterrows()
    ]


UFS_DISPONIVEIS = sorted(
    set(
        estacoes_geo[
            "mun_uf"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )
    |
    set(
        ANA[
            "uf"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )
)

UFS_DISPONIVEIS = [
    uf
    for uf
    in UFS_DISPONIVEIS
    if uf
]


# =============================================================================
# DROPDOWN ÚNICO DE ESTAÇÕES
# =============================================================================

opcoes_estacao = [

    {

        "label":
            (
                f"{linha['nome']} "
                f"({linha['codigo_exibicao']})"
            ),

        "value":
            linha[
                "codigo"
            ],
    }

    for _, linha
    in estacoes_geo
    .sort_values(
        [
            "mun_uf",
            "nome",
        ]
    )
    .iterrows()
]


ESTACAO_INICIAL = (

    "15630000"

    if (
        "15630000"
        in
        estacoes_geo[
            "codigo"
        ].astype(str).values
    )

    else

    estacoes_geo[
        "codigo"
    ].iloc[0]
)


# =============================================================================
# DASH
# =============================================================================

app = Dash(
    __name__
)

# Servidor Flask exposto para o Gunicorn/Render.
server = app.server


app.title = (
    "Condição Hidrológica e Saúde"
)


# =============================================================================
# LAYOUT
# =============================================================================

app.layout = html.Div(

    style={

        "fontFamily":
            "Arial, sans-serif",

        "backgroundColor":
            "#f4f6f8",

        "minHeight":
            "100vh",

        "padding":
            "18px",
    },

    children=[


        html.H2(

            "Condição Hidrológica Aplicada à Saúde",

            style={
                "marginBottom":
                    "2px",
            },
        ),


        html.Div(

            (
                "Monitoramento hidrológico voltado para à Saúde"
            ),

            style={

                "color":
                    "#666",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # MAPA NACIONAL
        # =====================================================================

        html.Div(

            [

                html.H3(
                    "Condição hidrológica das estações"
                ),

                html.Div(

                    (
                        "Clique em uma estação para realizar "
                        "a análise do entorno."
                    ),

                    style={
                        "fontSize":
                            "13px",

                        "color":
                            "#666",
                    },
                ),

                html.Div(
                    [
                        html.B("Camadas adicionais:"),
                        dcc.Checklist(
                            id="camadas-mapa-nacional",
                            options=[
                                {"label": " Municípios", "value": "municipios"},
                                {"label": " Polos Base", "value": "polos"},
                                {"label": " UBS", "value": "ubs"},
                                {"label": " UBSI", "value": "ubsi"},
                                {"label": " UPA", "value": "upa"},
                            ],
                            value=[],
                            inline=True,
                            inputStyle={"marginLeft": "12px", "marginRight": "4px"},
                            style={"marginTop": "10px", "fontSize": "13px"},
                        ),
                    ]
                ),

                dcc.Graph(

                    id=
                        "mapa-nacional",

                    config={

                        "displaylogo":
                            False,

                        "scrollZoom":
                            True,
                    },
                ),

                html.Details(
                    [
                        html.Summary(
                            "Glossário das condições hidrológicas",
                            style={
                                "fontWeight": "bold",
                                "cursor": "pointer",
                                "fontSize": "13px",
                            },
                        ),

                        html.Div(
                            [
                                html.Div([
                                    html.B("Muito acima do esperado: "),
                                    "nível significativamente acima da referência histórica esperada para o período; não representa, por si só, alerta de inundação."
                                ]),
                                html.Div([
                                    html.B("Acima do esperado: "),
                                    "nível acima da referência histórica esperada, porém em condição menos intensa que a classe Muito acima do esperado."
                                ]),
                                html.Div([
                                    html.B("Dentro do esperado: "),
                                    "nível próximo à faixa histórica esperada para o período."
                                ]),
                                html.Div([
                                    html.B("Abaixo do esperado: "),
                                    "nível abaixo da referência histórica esperada, porém em condição menos intensa que a classe Muito abaixo do esperado."
                                ]),
                                html.Div([
                                    html.B("Muito abaixo do esperado: "),
                                    "nível significativamente abaixo da referência histórica esperada para o período."
                                ]),
                                html.Div([
                                    html.B("Sem classificação: "),
                                    "estação sem referência histórica suficiente para interpretar a cota atual."
                                ]),
                                html.Div([
                                    html.B("Atenção — cheia / Alerta — cheia / Inundação / Atenção — estiagem: "),
                                    "condições definidas a partir de limiares operacionais oficiais disponíveis para a estação."
                                ]),
                                html.Div(
                                    "Nas estações ANA, a comparação climatológica utiliza a referência histórica mensal calculada no painel. "
                                    "Nas estações CENSIPAM, quando existem limiares operacionais oficiais, eles têm prioridade sobre a comparação climatológica.",
                                    style={
                                        "marginTop": "8px",
                                        "fontSize": "11px",
                                        "color": "#666",
                                    },
                                ),
                            ],
                            style={
                                "marginTop": "10px",
                                "lineHeight": "1.5",
                                "fontSize": "12px",
                            },
                        ),
                    ],
                    style={
                        "marginTop": "8px",
                        "padding": "8px 10px",
                        "backgroundColor": "#f8f9fa",
                        "border": "1px solid #e5e5e5",
                        "borderRadius": "6px",
                    },
                ),
            ],

            style={

                "backgroundColor":
                    "white",

                "padding":
                    "10px",

                "borderRadius":
                    "8px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # CONTROLES
        # =====================================================================

        html.Div(

            [

                html.Div(

                    [

                        html.B(
                            "Estação fluviométrica"
                        ),

                        dcc.Dropdown(

                            id=
                                "estacao",

                            options=
                                opcoes_estacao,

                            value=
                                ESTACAO_INICIAL,

                            clearable=
                                False,

                            searchable=
                                True,
                        ),
                    ]
                ),


                html.Div(

                    [

                        html.B(
                            "Raio de análise"
                        ),

                        dcc.Dropdown(

                            id=
                                "raio",

                            options=[

                                {
                                    "label":
                                        "5 km",

                                    "value":
                                        5,
                                },

                                {
                                    "label":
                                        "10 km",

                                    "value":
                                        10,
                                },

                                {
                                    "label":
                                        "25 km",

                                    "value":
                                        25,
                                },

                                {
                                    "label":
                                        "50 km",

                                    "value":
                                        50,
                                },

                                {
                                    "label":
                                        "100 km",

                                    "value":
                                        100,
                                },
                            ],

                            value=
                                25,

                            clearable=
                                False,
                        ),
                    ]
                ),
            ],

            style={

                "display":
                    "grid",

                "gridTemplateColumns":
                    "4fr 1fr",

                "gap":
                    "15px",

                "backgroundColor":
                    "white",

                "padding":
                    "15px",

                "borderRadius":
                    "8px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # CARDS ESTAÇÃO
        # =====================================================================

        html.Div(

            id=
                "cards-estacao",

            style={

                "display":
                    "grid",

                "gridTemplateColumns":
                    "repeat(5,1fr)",

                "gap":
                    "12px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # DESCRIÇÃO
        # =====================================================================

        html.Div(

            id=
                "descricao-condicao",

            style={

                "backgroundColor":
                    "white",

                "padding":
                    "15px",

                "borderRadius":
                    "8px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # CLIMATOLOGIA
        # =====================================================================

        html.Div(

            id=
                "cards-climatologia",

            style={

                "display":
                    "grid",

                "gridTemplateColumns":
                    "repeat(5,1fr)",

                "gap":
                    "12px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # SAÚDE
        # =====================================================================

        html.Div(

            id=
                "cards-entorno",

            style={

                "display":
                    "grid",

                "gridTemplateColumns":
                    "repeat(4,1fr)",

                "gap":
                    "12px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # INFO HIDROGRAFIA
        # =====================================================================

        html.Div(

            id=
                "info-rede",

            style={

                "backgroundColor":
                    "white",

                "padding":
                    "15px",

                "borderRadius":
                    "8px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # MAPA LOCAL
        # =====================================================================

        html.Div(

            dcc.Graph(

                id=
                    "mapa-local",

                config={

                    "displaylogo":
                        False,

                    "scrollZoom":
                        True,
                },
            ),

            style={

                "backgroundColor":
                    "white",

                "padding":
                    "5px",

                "borderRadius":
                    "8px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # LIMIARES
        # =====================================================================

        html.Div(

            [

                html.H3(
                    "Limiares oficiais da estação"
                ),

                dash_table.DataTable(

                    id=
                        "tabela-limiares",

                    columns=[

                        {
                            "name":
                                "Limiar",

                            "id":
                                "Limiar",
                        },

                        {
                            "name":
                                "Valor",

                            "id":
                                "Valor",
                        },

                        {
                            "name":
                                "Situação",

                            "id":
                                "Situação",
                        },
                    ],

                    style_cell={

                        "textAlign":
                            "left",

                        "padding":
                            "8px",

                        "fontSize":
                            "13px",
                    },

                    style_header={

                        "fontWeight":
                            "bold",

                        "backgroundColor":
                            "#f0f0f0",
                    },

                    style_data_conditional=[

                        {

                            "if": {

                                "filter_query":
                                    '{Situação} = "ACIONADO"'
                            },

                            "backgroundColor":
                                "#fff3cd",

                            "fontWeight":
                                "bold",
                        },
                    ],
                ),
            ],

            style={

                "backgroundColor":
                    "white",

                "padding":
                    "15px",

                "borderRadius":
                    "8px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # UNIDADES
        # =====================================================================

        html.Div(

            [

                html.H3(
                    id=
                        "titulo-unidades"
                ),

                dash_table.DataTable(

                    id=
                        "tabela-unidades",

                    columns=[

                        {
                            "name":
                                "Tipo",

                            "id":
                                "Tipo",
                        },

                        {
                            "name":
                                "Unidade",

                            "id":
                                "Unidade",
                        },

                        {
                            "name":
                                "CNES",

                            "id":
                                "CNES",
                        },

                        {
                            "name":
                                "Município",

                            "id":
                                "Município",
                        },

                        {
                            "name":
                                "UF",

                            "id":
                                "UF",
                        },

                        {
                            "name":
                                "Polo Base",

                            "id":
                                "Polo Base",
                        },

                        {
                            "name":
                                "DSEI",

                            "id":
                                "DSEI",
                        },

                        {
                            "name":
                                "Distância da estação (km)",

                            "id":
                                "Distância da estação (km)",
                        },
                    ],

                    page_size=
                        15,

                    sort_action=
                        "native",

                    filter_action=
                        "native",

                    style_table={
                        "overflowX":
                            "auto",
                    },

                    style_cell={

                        "textAlign":
                            "left",

                        "padding":
                            "7px",

                        "fontSize":
                            "12px",

                        "whiteSpace":
                            "normal",
                    },

                    style_header={

                        "fontWeight":
                            "bold",

                        "backgroundColor":
                            "#f0f0f0",
                    },
                ),
            ],

            style={

                "backgroundColor":
                    "white",

                "padding":
                    "15px",

                "borderRadius":
                    "8px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # POLOS
        # =====================================================================

        html.Div(

            [

                html.H3(
                    id=
                        "titulo-polos"
                ),

                dash_table.DataTable(

                    id=
                        "tabela-polos",

                    columns=[

                        {
                            "name":
                                "Polo Base",

                            "id":
                                "Polo Base",
                        },

                        {
                            "name":
                                "DSEI",

                            "id":
                                "DSEI",
                        },

                        {
                            "name":
                                "UF",

                            "id":
                                "UF",
                        },

                        {
                            "name":
                                "Distância da estação (km)",

                            "id":
                                "Distância da estação (km)",
                        },
                    ],

                    page_size=
                        10,

                    sort_action=
                        "native",

                    style_cell={

                        "textAlign":
                            "left",

                        "padding":
                            "7px",

                        "fontSize":
                            "12px",
                    },

                    style_header={

                        "fontWeight":
                            "bold",

                        "backgroundColor":
                            "#f0f0f0",
                    },
                ),
            ],

            style={

                "backgroundColor":
                    "white",

                "padding":
                    "15px",

                "borderRadius":
                    "8px",

                "marginBottom":
                    "15px",
            },
        ),


        # =====================================================================
        # COTAGRAMA
        # =====================================================================

        html.Div(

            [

                html.H3(
                    "Cotagrama"
                ),

                html.Div(

                    [

                        html.B(
                            "Período: "
                        ),

                        dcc.RadioItems(

                            id=
                                "periodo",

                            options=[

                                {
                                    "label":
                                        "30 dias",

                                    "value":
                                        "30d",
                                },

                                {
                                    "label":
                                        "3 meses",

                                    "value":
                                        "3m",
                                },

                                {
                                    "label":
                                        "6 meses",

                                    "value":
                                        "6m",
                                },

                                {
                                    "label":
                                        "1 ano",

                                    "value":
                                        "1a",
                                },

                                {
                                    "label":
                                        "Tudo",

                                    "value":
                                        "tudo",
                                },
                            ],

                            value=
                                "1a",

                            inline=
                                True,
                        ),
                    ]
                ),

                dcc.Graph(
                    id=
                        "cotagrama"
                ),
            ],

            style={

                "backgroundColor":
                    "white",

                "padding":
                    "15px",

                "borderRadius":
                    "8px",
            },
        ),


        # =====================================================================
        # PREVISÃO SAZONAL CEMADEN
        # =====================================================================

        html.Div(

            [

                html.H3(
                    "Previsão de Vazão Sazonal — Setembro/2026"
                ),

                html.Div(
                    PREVISAO_CEMADEN_RESUMO,
                    style={
                        "fontSize":
                            "13px",
                        "lineHeight":
                            "1.5",
                        "marginBottom":
                            "12px",
                    },
                ),

                (
                    html.Img(
                        src=
                            PREVISAO_CEMADEN_IMG,
                        style={
                            "display":
                                "block",
                            "maxWidth":
                                "520px",
                            "width":
                                "100%",
                            "height":
                                "auto",
                            "margin":
                                "0 auto",
                            "border":
                                "1px solid #ddd",
                        },
                    )
                    if PREVISAO_CEMADEN_IMG
                    else
                    html.Div(
                        (
                            "Figura não encontrada em: "
                            +
                            str(
                                ARQ_PREVISAO_CEMADEN
                            )
                        ),
                        style={
                            "padding":
                                "12px",
                            "backgroundColor":
                                "#fff3cd",
                            "border":
                                "1px solid #ffe69c",
                            "borderRadius":
                                "6px",
                            "fontSize":
                                "12px",
                        },
                    )
                ),

                html.Div(
                    "Fonte: Cemaden, 2026.",
                    style={
                        "fontSize":
                            "11px",
                        "color":
                            "#666",
                        "marginTop":
                            "10px",
                        "textAlign":
                            "center",
                    },
                ),
            ],

            style={
                "backgroundColor":
                    "white",
                "padding":
                    "15px",
                "borderRadius":
                    "8px",
                "marginTop":
                    "15px",
            },
        ),

        # =====================================================================
        # RESUMO ESTADUAL
        # =====================================================================

        html.Div(

            [

                html.H3(
                    "Resumo hidrológico por estado"
                ),

                html.Div(
                    (
                        "Síntese automática das estações monitoradas no estado, "
                        "combinando CENSIPAM e ANA/SNIRH. No CENSIPAM são preservados "
                        "os limiares e referências do SipamHidro; para a ANA, a cota "
                        "atual é comparada à referência histórica mensal calculada "
                        "a partir das médias mensais anuais com dados válidos."
                    ),
                    style={
                        "fontSize":
                            "12px",
                        "color":
                            "#666",
                        "marginBottom":
                            "10px",
                    },
                ),

                dcc.Dropdown(
                    id=
                        "relatorio-uf",
                    options=
                        [
                            {
                                "label":
                                    uf,
                                "value":
                                    uf,
                            }
                            for uf
                            in UFS_DISPONIVEIS
                        ],
                    value=
                        (
                            "AM"
                            if "AM" in UFS_DISPONIVEIS
                            else (
                                UFS_DISPONIVEIS[0]
                                if UFS_DISPONIVEIS
                                else None
                            )
                        ),
                    clearable=
                        False,
                    style={
                        "maxWidth":
                            "300px",
                        "marginBottom":
                            "12px",
                    },
                ),

                html.Div(
                    id=
                        "relatorio-estado"
                ),
            ],

            style={
                "backgroundColor":
                    "white",
                "padding":
                    "15px",
                "borderRadius":
                    "8px",
                "marginTop":
                    "15px",
            },
        ),


        html.Div(
            [
                html.B("Fontes"),
                html.Div("Dados fluviométricos: redes de monitoramento hidrológico CENSIPAM e ANA/SNIRH."),
                html.Div("Estabelecimentos de saúde: IDE, CNES."),
                html.Div("Polos Base: SESAI."),
                html.Div("Trechos dos rios: ANA."),
                html.Div("Previsão sazonal de vazão: Cemaden."),
                html.Div(
                    "Desenvolvido por: CGClima/DVSAT/SVSA.",
                    style={"fontWeight": "bold", "marginTop": "5px"},
                ),
            ],
            style={
                "backgroundColor": "white", "padding": "12px 15px",
                "borderRadius": "8px", "marginTop": "15px",
                "fontSize": "12px", "color": "#555", "lineHeight": "1.45",
            },
        ),
    ],
)


# =============================================================================
# MAPA NACIONAL
# =============================================================================

@app.callback(
    Output("mapa-nacional", "figure"),
    Input("estacao", "value"),
    Input("camadas-mapa-nacional", "value"),
)
def atualizar_mapa_nacional(codigo_selecionado, camadas_ativas):

    camadas_ativas = camadas_ativas or []
    fig = go.Figure()

    # Camada opcional: municípios
    if "municipios" in camadas_ativas:
        lon, lat = geometrias_para_linhas(MUN)
        fig.add_trace(
            go.Scattergeo(
                lon=lon,
                lat=lat,
                mode="lines",
                name="Municípios",
                line={
                    "width": 0.45,
                    "color": "#bdbdbd",
                },
                hoverinfo="skip",
            )
        )

    # Camada opcional: Polos Base
    # POLOS é uma camada poligonal, portanto deve ser desenhada como linhas.
    if "polos" in camadas_ativas:

        temp_polos = POLOS[
            POLOS.geometry.notna()
            &
            (~POLOS.geometry.is_empty)
        ].copy()

        lon_polos, lat_polos = geometrias_para_linhas(
            temp_polos
        )

        fig.add_trace(
            go.Scattergeo(
                lon=lon_polos,
                lat=lat_polos,
                mode="lines",
                name="Polos Base",
                line={
                    "width": 1.2,
                    "color": "#A0A0A0",
                },
                hoverinfo="skip",
            )
        )

    # Camadas opcionais: estabelecimentos
    for chave, tipo, cor, simbolo in [
        ("ubs", "UBS", "#3182bd", "circle"),
        ("ubsi", "UBSI", "#9b59b6", "square"),
        ("upa", "UPA", "#e67e22", "diamond"),
    ]:
        if chave not in camadas_ativas:
            continue

        temp = SAUDE[
            SAUDE["tipo"] == tipo
        ].copy()

        if temp.empty:
            continue

        fig.add_trace(
            go.Scattergeo(
                lon=temp.geometry.x,
                lat=temp.geometry.y,
                mode="markers",
                name=tipo,
                marker={
                    "size": 5.5,
                    "symbol": simbolo,
                    "color": cor,
                    "opacity": .75,
                    "line": {
                        "width": .35,
                        "color": "white",
                    },
                },
                customdata=np.column_stack([
                    temp["nome"],
                    temp["cnes"],
                    temp["municipio"],
                    temp["uf"],
                    temp["polo_base"],
                    temp["dsei"],
                ]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>CNES: %{customdata[1]}"
                    "<br>Município: %{customdata[2]}/%{customdata[3]}"
                    "<br>Polo Base: %{customdata[4]}"
                    "<br>DSEI: %{customdata[5]}"
                    "<extra></extra>"
                ),
            )
        )


    # Estações ficam sempre visíveis e clicáveis.
    grupos = {}

    for idx, linha in estacoes_geo.iterrows():

        # Exibe no mapa somente estações com condição hidrológica classificável.
        if str(linha["condicao_hidrologica"]).strip().upper() == "SEM CLASSIFICAÇÃO":
            continue

        simbolo, cor = estilo_estacao(
            linha["condicao_hidrologica"],
            linha["tendencia"],
        )

        chave = (
            linha["condicao_hidrologica"],
            simbolo,
            cor,
        )

        grupos.setdefault(
            chave,
            [],
        ).append(
            idx
        )

    for (condicao, simbolo, cor), indices in grupos.items():

        temp = estacoes_geo.loc[
            indices
        ].copy()

        tamanhos = np.where(
            temp["codigo"] == str(codigo_selecionado),
            17,
            9,
        )

        fig.add_trace(
            go.Scattergeo(
                lon=temp["longitude"],
                lat=temp["latitude"],
                mode="markers",
                name=condicao.title(),
                marker={
                    "size": tamanhos,
                    "symbol": simbolo,
                    "color": cor,
                    "opacity": .96,
                    "line": {
                        "width": .7,
                        "color": "white",
                    },
                },
                customdata=np.column_stack([
                    temp["codigo"],
                    temp["nome"],
                    temp["cotaUltimaMedicao_m"],
                    temp["condicao_hidrologica"],
                    temp["tendencia"],
                    temp["delta_7d"],
                    temp["mun_nome"],
                    temp["mun_uf"],
                    temp["codigo_exibicao"],
                ]),
                hovertemplate=(
                    "<b>%{customdata[1]}</b>"
                    "<br>Código: %{customdata[8]}"
                    "<br>Cota atual: %{customdata[2]:.2f} m"
                    "<br><b>Condição hidrológica:</b> %{customdata[3]}"
                    "<br>Município: %{customdata[6]}/%{customdata[7]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_geos(
        projection_type="mercator",
        lonaxis_range=[-75, -30],
        lataxis_range=[-35, 7],
        showland=True,
        landcolor="#fafafa",
        showocean=True,
        oceancolor="#eaf2f8",
        showlakes=True,
        lakecolor="#eaf2f8",
        showcountries=True,
        countrycolor="#777777",
        showcoastlines=True,
        coastlinecolor="#777777",
        bgcolor="white",
    )

    fig.update_layout(
        title="Brasil — condição hidrológica das estações monitoradas",
        height=700,
        margin={
            "t": 50,
            "l": 5,
            "r": 5,
            "b": 5,
        },
        legend={
            "x": .01,
            "y": .99,
            "bgcolor": "rgba(255,255,255,0.90)",
        },
        uirevision="mapa-nacional",
    )

    return fig


# =============================================================================
# CLIQUE NA ESTAÇÃO
# =============================================================================

@app.callback(

    Output(
        "estacao",
        "value",
    ),

    Input(
        "mapa-nacional",
        "clickData",
    ),

    State(
        "estacao",
        "value",
    ),

    prevent_initial_call=
        True,
)

def selecionar_estacao_mapa(
    click_data,
    atual,
):

    if not click_data:

        return atual


    try:

        codigo = str(

            click_data[
                "points"
            ][0][
                "customdata"
            ][0]
        )

        return codigo

    except Exception:

        return atual


# =============================================================================
# CALLBACK LOCAL
# =============================================================================

@app.callback(

    Output(
        "cards-estacao",
        "children",
    ),

    Output(
        "descricao-condicao",
        "children",
    ),

    Output(
        "cards-climatologia",
        "children",
    ),

    Output(
        "cards-entorno",
        "children",
    ),

    Output(
        "info-rede",
        "children",
    ),

    Output(
        "mapa-local",
        "figure",
    ),

    Output(
        "tabela-limiares",
        "data",
    ),

    Output(
        "tabela-unidades",
        "data",
    ),

    Output(
        "tabela-polos",
        "data",
    ),

    Output(
        "titulo-unidades",
        "children",
    ),

    Output(
        "titulo-polos",
        "children",
    ),

    Input(
        "estacao",
        "value",
    ),

    Input(
        "raio",
        "value",
    ),
)

def atualizar_analise(
    codigo,
    raio_km,
):

    meta = estacoes_geo[
        estacoes_geo[
            "codigo"
        ]
        ==
        str(
            codigo
        )
    ]


    if meta.empty:

        return (
            [],
            "",
            [],
            [],
            "",
            go.Figure(),
            [],
            [],
            [],
            "",
            "",
        )


    linha = meta.iloc[
        0
    ]


    resultado = classificar_condicao_hidrologica(
        linha
    )


    condicao = resultado[
        "condicao"
    ]


    cor_condicao = CORES_CONDICAO.get(
        condicao,
        "#777777",
    )


    # =========================================================================
    # SAÚDE
    # =========================================================================

    unidades_proj = unidades_no_entorno(
        codigo,
        raio_km,
    )


    tabela_saude = tabela_unidades(
        codigo,
        raio_km,
    )


    polos = polos_no_entorno(
        codigo,
        raio_km,
    )


    if unidades_proj.empty:

        n_ubs = 0
        n_ubsi = 0
        n_upa = 0

    else:

        n_ubs = int(
            (
                unidades_proj[
                    "tipo"
                ]
                ==
                "UBS"
            ).sum()
        )

        n_ubsi = int(
            (
                unidades_proj[
                    "tipo"
                ]
                ==
                "UBSI"
            ).sum()
        )

        n_upa = int(
            (
                unidades_proj[
                    "tipo"
                ]
                ==
                "UPA"
            ).sum()
        )


    # =========================================================================
    # HIDROGRAFIA
    # =========================================================================

    bho_local, rede, info_rede = (
        identificar_rede_conectada(

            codigo,

            raio_km,
        )
    )


    # =========================================================================
    # CARDS ESTAÇÃO
    # =========================================================================

    cards_estacao = [

        criar_card(

            "Condição hidrológica",

            condicao,

            resultado[
                "fonte"
            ],

            cor_borda=
                cor_condicao,
        ),


        criar_card(

            "Cota atual",

            (
                fmt_numero(
                    linha.get(
                        "cotaUltimaMedicao_m",
                        np.nan,
                    )
                )
                +
                " m"
            ),

            (
                "Última medição: "
                +
                fmt_data(
                    linha.get(
                        "dataHoraUltimaMedicao",
                        pd.NaT,
                    )
                )
            ),
        ),


        criar_card(

            "Tendência recente",

            resultado[
                "tendencia"
            ],

            (
                "Sem série recente contínua para tendência."
                if str(codigo).startswith("ANA:")
                else
                (
                    "Variação em 7 dias: "
                    +
                    fmt_numero(
                        resultado[
                            "delta_7d"
                        ]
                    )
                    +
                    " m"
                )
            ),
        ),


        criar_card(

            "Município",

            (
                texto_valido(
                    linha.get(
                        "mun_nome",
                        ""
                    )
                )
                or
                "—"
            ),

            texto_valido(
                linha.get(
                    "mun_uf",
                    ""
                )
            ),
        ),


        criar_card(

            "Polo Base / DSEI",

            (
                texto_valido(
                    linha.get(
                        "polo_nome",
                        ""
                    )
                )
                or
                "—"
            ),

            (
                texto_valido(
                    linha.get(
                        "dsei_nome",
                        ""
                    )
                )
                or
                "—"
            ),
        ),
    ]


    # =========================================================================
    # DESCRIÇÃO
    # =========================================================================

    descricao = [

        html.B(
            f"{linha['nome']} ({codigo})"
        ),

        html.Br(),

        html.Span(
            resultado[
                "descricao"
            ]
        ),

        html.Br(),

        html.Small(
            (
                "A condição apresentada refere-se à estação monitorada "
                "e não deve ser automaticamente extrapolada para todos "
                "os cursos d'água exibidos no entorno."
            )
        ),
    ]


    # =========================================================================
    # CLIMATOLOGIA
    # =========================================================================

    cards_climatologia = [

        criar_card(

            "Mínima histórica",

            (
                fmt_numero(
                    resultado[
                        "minimo"
                    ]
                )
                +
                " m"
            ),
        ),


        criar_card(

            "Média histórica",

            (
                fmt_numero(
                    resultado[
                        "media"
                    ]
                )
                +
                " m"
            ),
        ),


        criar_card(

            "Máxima histórica",

            (
                fmt_numero(
                    resultado[
                        "maximo"
                    ]
                )
                +
                " m"
            ),
        ),


        criar_card(

            "Variação em 24 h",

            (
                fmt_numero(
                    resultado[
                        "delta_1d"
                    ]
                )
                +
                " m"
            ),
        ),


        criar_card(

            "Variação em 7 dias",

            (
                fmt_numero(
                    resultado[
                        "delta_7d"
                    ]
                )
                +
                " m"
            ),
        ),
    ]



    # Para estações ANA, apresenta a referência histórica do mês
    # correspondente à observação atual.
    if str(
        codigo
    ).startswith(
        "ANA:"
    ):

        n_anos_txt = "—"

        if pd.notna(
            linha.get(
                "n_anos_validos",
                np.nan,
            )
        ):

            n_anos_txt = str(
                int(
                    float(
                        linha.get(
                            "n_anos_validos"
                        )
                    )
                )
            )

        cobertura_txt = "—"

        if pd.notna(
            linha.get(
                "cobertura_media",
                np.nan,
            )
        ):

            cobertura_txt = (
                f"{float(linha.get('cobertura_media')) * 100:.0f}%"
            )

        cards_climatologia = [

            criar_card(
                "Menor média mensal",
                fmt_numero(
                    linha.get(
                        "menor_media_mensal_m",
                        np.nan,
                    )
                )
                +
                " m",
            ),

            criar_card(
                "Média histórica mensal",
                fmt_numero(
                    linha.get(
                        "media_mensal_historica_m",
                        np.nan,
                    )
                )
                +
                " m",
            ),

            criar_card(
                "Maior média mensal",
                fmt_numero(
                    linha.get(
                        "maior_media_mensal_m",
                        np.nan,
                    )
                )
                +
                " m",
            ),

            criar_card(
                "Anos válidos",
                n_anos_txt,
                subtitulo=
                    texto_valido(
                        linha.get(
                            "periodo_referencia",
                            "",
                        )
                    ),
            ),

            criar_card(
                "Cobertura média",
                cobertura_txt,
                subtitulo=
                    (
                        "Mês de referência: "
                        +
                        texto_valido(
                            linha.get(
                                "nome_mes",
                                "",
                            )
                        )
                    ),
            ),
        ]


    # =========================================================================
    # CARDS SAÚDE
    # =========================================================================

    cards_entorno = [

        criar_card(

            f"UBS em até {raio_km} km",

            f"{n_ubs:,}",
        ),


        criar_card(

            f"UBSI em até {raio_km} km",

            f"{n_ubsi:,}",
        ),


        criar_card(

            f"UPA em até {raio_km} km",

            f"{n_upa:,}",
        ),


        criar_card(

            f"Polos Base em até {raio_km} km",

            f"{len(polos):,}",
        ),
    ]


    # =========================================================================
    # INFO REDE
    # =========================================================================

    if info_rede is None:

        bloco_rede = [

            html.H3(
                "Rede hidrográfica associada"
            ),

            html.Div(
                (
                    "Não foi possível associar a estação a um "
                    "trecho da BHO250 neste raio."
                )
            ),
        ]


    else:

        bloco_rede = [

            html.H3(
                "Rede hidrográfica associada"
            ),

            html.B(
                info_rede[
                    "nome_rio"
                ]
            ),

            html.Br(),

            html.Span(
                (
                    "Rio: "
                    f"{info_rede['nome_rio']}"
                )
            ),

            html.Br(),

            html.Span(
                (
                    "Bacia hidrográfica: "
                    f"{info_rede['nome_bacia']}"
                )
                if info_rede.get("nome_bacia")
                else
                "Bacia hidrográfica: nome não disponível na camada BHO250"
            ),

            html.Br(),

            html.Span(
                (
                    "Distância entre a estação e o trecho associado: "
                    f"{fmt_numero(info_rede['distancia_m'], 0)} m"
                )
            ),

            html.Br(),

            html.Span(
                (
                    f"Rede conectada identificada dentro do raio de "
                    f"{raio_km} km: {info_rede['n_total']:,} trechos."
                )
            ),

            html.Br(),

            html.Small(
                (
                    f"Montante: {info_rede['n_montante']:,} trechos | "
                    f"Jusante: {info_rede['n_jusante']:,} trechos. "
                    "A separação é derivada da topologia "
                    
                )
            ),
        ]


    # =========================================================================
    # MAPA LOCAL
    # =========================================================================

    fig = go.Figure()


    lat0 = float(
        linha[
            "latitude"
        ]
    )

    lon0 = float(
        linha[
            "longitude"
        ]
    )


    est_proj = ESTACOES_PROJ[
        ESTACOES_PROJ[
            "codigo"
        ]
        ==
        str(
            codigo
        )
    ]


    ponto_proj = est_proj.geometry.iloc[
        0
    ]


    buffer_proj = ponto_proj.buffer(
        float(
            raio_km
        )
        *
        1000
    )


    buffer_geo = (
        gpd.GeoSeries(

            [
                buffer_proj
            ],

            crs=
                CRS_METRICO,
        )
        .to_crs(
            CRS_GEO
        )
        .iloc[
            0
        ]
    )


    # =========================================================================
    # MUNICÍPIOS
    # =========================================================================

    ids_mun = MUN_PROJ.sindex.query(

        buffer_proj,

        predicate=
            "intersects",
    )


    if len(
        ids_mun
    ) > 0:

        mun_local = MUN.iloc[
            ids_mun
        ]


        lon_mun, lat_mun = (
            geometrias_para_linhas(
                mun_local
            )
        )


        fig.add_trace(

            go.Scattergeo(

                lon=
                    lon_mun,

                lat=
                    lat_mun,

                mode=
                    "lines",

                name=
                    "Municípios",

                line={

                    "width":
                        0.6,

                    "color":
                        "#bdbdbd",
                },

                hoverinfo=
                    "skip",
            )
        )


    # =========================================================================
    # POLOS
    # =========================================================================

    ids_polos = POLOS_PROJ.sindex.query(

        buffer_proj,

        predicate=
            "intersects",
    )


    if len(
        ids_polos
    ) > 0:

        polos_local = POLOS.iloc[
            ids_polos
        ]


        lon_polo, lat_polo = (
            geometrias_para_linhas(
                polos_local
            )
        )


        fig.add_trace(

            go.Scattergeo(

                lon=
                    lon_polo,

                lat=
                    lat_polo,

                mode=
                    "lines",

                name=
                    "Polos Base",

                line={

                    "width":
                        1.2,

                    "color":
                        "#238b45",
                },

                hoverinfo=
                    "skip",
            )
        )


    # =========================================================================
    # TODOS OS RIOS DO RAIO
    # =========================================================================

    if not bho_local.empty:

        outros = bho_local[
            ~bho_local.index.isin(
                rede.index
            )
        ].copy()


        if not outros.empty:

            lon_rio, lat_rio = (
                geometrias_para_linhas(
                    outros
                )
            )


            fig.add_trace(

                go.Scattergeo(

                    lon=
                        lon_rio,

                    lat=
                        lat_rio,

                    mode=
                        "lines",

                    name=
                        "Cursos d'água",

                    line={

                        "width":
                            0.7,

                        "color":
                            "#9ecae1",
                    },

                    hoverinfo=
                        "skip",
                )
            )


    # =========================================================================
    # REDE CONECTADA
    # =========================================================================

    if not rede.empty:

        montante = rede[
            rede[
                "relacao_rede"
            ]
            ==
            "Montante"
        ]


        if not montante.empty:

            lon_mont, lat_mont = (
                geometrias_para_linhas(
                    montante
                )
            )


            fig.add_trace(

                go.Scattergeo(

                    lon=
                        lon_mont,

                    lat=
                        lat_mont,

                    mode=
                        "lines",

                    name=
                        "Rede conectada — montante",

                    line={

                        "width":
                            2.0,

                        "color":
                            "#3182bd",
                    },

                    hoverinfo=
                        "skip",
                )
            )


        jusante = rede[
            rede[
                "relacao_rede"
            ]
            ==
            "Jusante"
        ]


        if not jusante.empty:

            lon_jus, lat_jus = (
                geometrias_para_linhas(
                    jusante
                )
            )


            fig.add_trace(

                go.Scattergeo(

                    lon=
                        lon_jus,

                    lat=
                        lat_jus,

                    mode=
                        "lines",

                    name=
                        "Rede conectada — jusante",

                    line={

                        "width":
                            2.3,

                        "color":
                            "#08519c",
                    },

                    hoverinfo=
                        "skip",
                )
            )


        trecho = rede[
            rede[
                "relacao_rede"
            ]
            ==
            "Trecho da estação"
        ]


        if not trecho.empty:

            lon_tr, lat_tr = (
                geometrias_para_linhas(
                    trecho
                )
            )


            fig.add_trace(

                go.Scattergeo(

                    lon=
                        lon_tr,

                    lat=
                        lat_tr,

                    mode=
                        "lines",

                    name=
                        "Trecho associado à estação",

                    line={

                        "width":
                            4,

                        "color":
                            "#08306b",
                    },

                    hoverinfo=
                        "skip",
                )
            )


    # =========================================================================
    # RAIO
    # =========================================================================

    lon_buffer = []

    lat_buffer = []


    if buffer_geo.geom_type == "Polygon":

        for lon, lat in (
            buffer_geo
            .exterior
            .coords
        ):

            lon_buffer.append(
                lon
            )

            lat_buffer.append(
                lat
            )


    fig.add_trace(

        go.Scattergeo(

            lon=
                lon_buffer,

            lat=
                lat_buffer,

            mode=
                "lines",

            name=
                f"Raio de análise — {raio_km} km",

            line={

                "width":
                    1.4,

                "dash":
                    "dash",

                "color":
                    "#555555",
            },

            hoverinfo=
                "skip",
        )
    )


    # =========================================================================
    # UNIDADES
    # =========================================================================

    if not unidades_proj.empty:

        unidades_geo = unidades_proj.to_crs(
            CRS_GEO
        )


        config_saude = {

            "UBS": {

                "symbol":
                    "circle",

                "color":
                    "#2171b5",

                "size":
                    7,
            },

            "UBSI": {

                "symbol":
                    "square",

                "color":
                    "#8e44ad",

                "size":
                    9,
            },

            "UPA": {

                "symbol":
                    "diamond",

                "color":
                    "#e6550d",

                "size":
                    10,
            },
        }


        for tipo, cfg in config_saude.items():

            temp = unidades_geo[
                unidades_geo[
                    "tipo"
                ]
                ==
                tipo
            ].copy()


            if temp.empty:

                continue


            fig.add_trace(

                go.Scattergeo(

                    lon=
                        temp.geometry.x,

                    lat=
                        temp.geometry.y,

                    mode=
                        "markers",

                    name=
                        tipo,

                    marker={

                        "size":
                            cfg[
                                "size"
                            ],

                        "symbol":
                            cfg[
                                "symbol"
                            ],

                        "color":
                            cfg[
                                "color"
                            ],

                        "opacity":
                            0.90,

                        "line": {

                            "width":
                                0.5,

                            "color":
                                "white",
                        },
                    },

                    customdata=np.column_stack(
                        [

                            temp[
                                "nome"
                            ],

                            temp[
                                "cnes"
                            ],

                            temp[
                                "municipio"
                            ],

                            temp[
                                "uf"
                            ],

                            temp[
                                "polo_base"
                            ],

                            temp[
                                "dsei"
                            ],

                            temp[
                                "dist_km"
                            ].round(
                                2
                            ),
                        ]
                    ),

                    hovertemplate=(

                        "<b>%{customdata[0]}</b>"

                        "<br>CNES: %{customdata[1]}"

                        "<br>Município: %{customdata[2]}"

                        "<br>UF: %{customdata[3]}"

                        "<br>Polo Base: %{customdata[4]}"

                        "<br>DSEI: %{customdata[5]}"

                        "<br>Distância da estação: "
                        "%{customdata[6]} km"

                        "<extra></extra>"
                    ),
                )
            )


    # =========================================================================
    # ESTAÇÃO
    # =========================================================================

    fig.add_trace(

        go.Scattergeo(

            lon=[
                lon0
            ],

            lat=[
                lat0
            ],

            mode=
                "markers",

            name=
                "Estação fluviométrica",

            marker={

                "size":
                    17,

                "symbol":
                    "triangle-up",

                "color":
                    cor_condicao,

                "line": {

                    "width":
                        1.5,

                    "color":
                        "black",
                },
            },

            hovertemplate=(

                f"<b>{linha['nome']}</b>"

                f"<br>Código: {codigo}"

                f"<br>Cota atual: "
                f"{fmt_numero(linha.get('cotaUltimaMedicao_m'))} m"

                f"<br>Condição: {condicao}"

                "<extra></extra>"
            ),
        )
    )


    # =========================================================================
    # EXTENSÃO DO MAPA
    # =========================================================================

    margem_lat = (
        float(
            raio_km
        )
        /
        111.0
        *
        1.25
    )


    cos_lat = max(

        math.cos(
            math.radians(
                lat0
            )
        ),

        0.20,
    )


    margem_lon = (
        float(
            raio_km
        )
        /
        (
            111.0
            *
            cos_lat
        )
        *
        1.25
    )


    fig.update_geos(

        projection_type=
            "mercator",

        lonaxis_range=[

            lon0
            -
            margem_lon,

            lon0
            +
            margem_lon,
        ],

        lataxis_range=[

            lat0
            -
            margem_lat,

            lat0
            +
            margem_lat,
        ],

        showland=
            True,

        landcolor=
            "#fafafa",

        showocean=
            True,

        oceancolor=
            "#eaf2f8",

        showlakes=
            True,

        lakecolor=
            "#eaf2f8",

        showcountries=
            True,

        countrycolor=
            "#777777",

        showcoastlines=
            True,

        coastlinecolor=
            "#777777",

        bgcolor=
            "white",
    )


    fig.update_layout(

        title=(
            f"Entorno da estação — {linha['nome']} | "
            f"raio de {raio_km} km"
        ),

        height=
            720,

        margin={
            "t": 50,
            "l": 5,
            "r": 5,
            "b": 5,
        },

        legend={

            "x":
                0.01,

            "y":
                0.99,

            "bgcolor":
                "rgba(255,255,255,0.90)",
        },

        uirevision=
            f"{codigo}-{raio_km}",
    )


    # =========================================================================
    # RETURN
    # =========================================================================

    return (

        cards_estacao,

        descricao,

        cards_climatologia,

        cards_entorno,

        bloco_rede,

        fig,

        tabela_limites_oficiais(
            linha
        ),

        tabela_saude.to_dict(
            "records"
        ),

        (
            polos.to_dict(
                "records"
            )
            if not polos.empty
            else []
        ),

        (
            f"Estabelecimentos de saúde em até "
            f"{raio_km} km ({len(tabela_saude):,})"
        ),

        (
            f"Polos Base em até "
            f"{raio_km} km ({len(polos):,})"
        ),
    )



# =============================================================================
# RELATÓRIO ESTADUAL
# =============================================================================

@app.callback(

    Output(
        "relatorio-estado",
        "children",
    ),

    Input(
        "relatorio-uf",
        "value",
    ),
)
def gerar_relatorio_estado(
    uf,
):

    if not uf:

        return html.Div(
            "Selecione uma UF."
        )


    dados = estacoes_geo[
        estacoes_geo[
            "mun_uf"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        ==
        str(
            uf
        ).upper()
    ].copy()


    if dados.empty:

        return html.Div(
            (
                "Não foram encontradas estações com cota atual "
                f"disponível para {uf}."
            )
        )


    n_estacoes = len(
        dados
    )

    n_municipios = (
        dados[
            "mun_nome"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace(
            "",
            np.nan,
        )
        .nunique()
    )


    cond = (
        dados[
            "condicao_hidrologica"
        ]
        .fillna(
            "SEM CLASSIFICAÇÃO"
        )
        .value_counts()
    )


    categorias_baixa = [
        "ATENÇÃO — ESTIAGEM",
        "MUITO ABAIXO DO ESPERADO",
        "ABAIXO DO ESPERADO",
    ]

    categorias_alta = [
        "ATENÇÃO — CHEIA",
        "ALERTA — CHEIA",
        "INUNDAÇÃO",
        "MUITO ACIMA DO ESPERADO",
        "ACIMA DO ESPERADO",
    ]


    n_baixa = int(
        dados[
            "condicao_hidrologica"
        ]
        .isin(
            categorias_baixa
        )
        .sum()
    )

    n_alta = int(
        dados[
            "condicao_hidrologica"
        ]
        .isin(
            categorias_alta
        )
        .sum()
    )

    n_monitoramento = int(
        (
            dados[
                "condicao_hidrologica"
            ]
            ==
            "DENTRO DO ESPERADO"
        ).sum()
    )

    n_sem_class = int(
        (
            dados[
                "condicao_hidrologica"
            ]
            ==
            "SEM CLASSIFICAÇÃO"
        ).sum()
    )


    destaque = dados[
        dados[
            "condicao_hidrologica"
        ]
        .isin(
            categorias_baixa
            +
            categorias_alta
        )
    ].copy()


    municipios_destaque = sorted(
        {
            texto_valido(
                x
            )
            for x
            in destaque[
                "mun_nome"
            ].tolist()
            if texto_valido(
                x
            )
        }
    )


    # Estruturas de saúde existentes nos municípios com estações destacadas.
    saude_uf = SAUDE[
        SAUDE[
            "uf"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        ==
        str(
            uf
        ).upper()
    ].copy()


    if municipios_destaque:

        _mun_norm = {
            m.upper().strip()
            for m
            in municipios_destaque
        }

        saude_destaque = saude_uf[
            saude_uf[
                "municipio"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
            .isin(
                _mun_norm
            )
        ].copy()

    else:

        saude_destaque = saude_uf.iloc[
            0:0
        ].copy()


    n_ubs = int(
        (
            saude_destaque[
                "tipo"
            ]
            ==
            "UBS"
        ).sum()
    ) if not saude_destaque.empty else 0

    n_ubsi = int(
        (
            saude_destaque[
                "tipo"
            ]
            ==
            "UBSI"
        ).sum()
    ) if not saude_destaque.empty else 0

    n_upa = int(
        (
            saude_destaque[
                "tipo"
            ]
            ==
            "UPA"
        ).sum()
    ) if not saude_destaque.empty else 0


    if str(
        uf
    ).upper() in UFS_PREVISAO_ATENCAO:

        previsao_estado = (
            "A perspectiva sazonal de vazão para setembro/2026 indica, "
            "na interpretação visual preliminar da figura do Cemaden, "
            "áreas do estado com sinal de vazões abaixo da média. "
            "Esse cenário não representa um alerta pontual, mas reforça "
            "a necessidade de acompanhar especialmente localidades que "
            "já apresentam cotas reduzidas ou outras condições hidrológicas "
            "desfavoráveis nas estações monitoradas."
        )

    else:

        previsao_estado = (
            "Na interpretação visual preliminar da perspectiva sazonal de "
            "setembro/2026, o estado não foi incluído entre aqueles com sinal "
            "predominante de vazões abaixo da média neste protótipo. "
            "O monitoramento observacional das estações permanece necessário, "
            "pois a previsão sazonal não substitui a condição medida localmente."
        )


    if municipios_destaque:

        locais = ", ".join(
            municipios_destaque[
                :25
            ]
        )

        if len(
            municipios_destaque
        ) > 25:

            locais += " e outras localidades"

        texto_localidades = (
            f"Foram identificadas {len(municipios_destaque)} localidades "
            "com pelo menos uma estação em condição de cota baixa, abaixo do "
            "esperado, elevada, acima do esperado ou em limiar operacional: "
            f"{locais}."
        )

    else:

        texto_localidades = (
            "No conjunto atualmente disponível, não foram identificadas "
            "localidades com estações em condição hidrológica destacada."
        )


    texto_saude = (
        f"Nos municípios destacados existem {n_ubs} UBS, {n_ubsi} UBSI "
        f"e {n_upa} UPA cadastradas na base do painel. "
        "A presença dessas estruturas no mesmo município não significa, por "
        "si só, exposição ou impacto; a avaliação operacional deve considerar "
        "a proximidade da unidade com a estação e a rede hidrográfica, que pode "
        "ser verificada selecionando cada estação no painel."
        if municipios_destaque
        else
        (
            "Sem localidades destacadas neste momento, recomenda-se manter "
            "o acompanhamento das medições e da perspectiva sazonal."
        )
    )


    cards = html.Div(

        [

            criar_card(
                "Estações com cota atual",
                f"{n_estacoes:,}",
            ),

            criar_card(
                "Municípios monitorados",
                f"{n_municipios:,}",
            ),

            criar_card(
                "Cota baixa / abaixo",
                f"{n_baixa:,}",
            ),

            criar_card(
                "Muito acima do esperado / acima",
                f"{n_alta:,}",
            ),

            criar_card(
                "Monitoramento",
                f"{n_monitoramento:,}",
            ),
        ],

        style={
            "display":
                "grid",
            "gridTemplateColumns":
                "repeat(5,1fr)",
            "gap":
                "10px",
            "marginBottom":
                "12px",
        },
    )


    tabela = dados.copy()

    tabela[
        "Estação"
    ] = tabela[
        "nome"
    ]

    tabela[
        "Rio"
    ] = tabela.get(
        "rio",
        ""
    ).fillna("")

    tabela[
        "Município"
    ] = tabela[
        "mun_nome"
    ].fillna("")

    tabela[
        "Cota atual (m)"
    ] = tabela[
        "cotaUltimaMedicao_m"
    ].apply(
        fmt_numero
    )

    tabela[
        "Condição"
    ] = tabela[
        "condicao_hidrologica"
    ].fillna(
        "SEM CLASSIFICAÇÃO"
    )

    tabela[
        "Último registro"
    ] = tabela[
        "dataHoraUltimaMedicao"
    ].apply(
        fmt_data
    )


    tabela = tabela[
        [
            "Estação",
            "Rio",
            "Município",
            "Cota atual (m)",
            "Condição",
            "Último registro",
        ]
    ].sort_values(
        [
            "Condição",
            "Município",
            "Estação",
        ]
    )


    return html.Div(

        [

            cards,

            html.Div(

                [

                    html.H4(
                        f"Síntese hidrológica — {uf}"
                    ),

                    html.P(
                        (
                            f"O estado possui {n_estacoes:,} estações com "
                            f"cota atual disponível, distribuídas em "
                            f"{n_municipios:,} municípios. "
                            f"Dessas, {n_baixa:,} apresentam condição de cota "
                            f"baixa ou abaixo do esperado e {n_alta:,} "
                            "apresentam cota elevada, acima do esperado ou "
                            "condição operacional relacionada à cheia."
                        )
                    ),

                    html.P(
                        texto_localidades
                    ),

                    html.P(
                        previsao_estado
                    ),

                    html.P(
                        texto_saude
                    ),

                    html.Small(
                        (
                            "Nota de interpretação: as condições das estações "
                            "representam o ponto monitorado e não devem ser "
                            "automaticamente extrapoladas para todo o município "
                            "ou para todos os cursos d'água próximos. A previsão "
                            "sazonal é contexto de planejamento e não substitui "
                            "as medições observadas."
                        )
                    ),
                ],

                style={
                    "backgroundColor":
                        "#f8fafc",
                    "padding":
                        "14px",
                    "borderRadius":
                        "7px",
                    "marginBottom":
                        "14px",
                    "lineHeight":
                        "1.5",
                },
            ),

            html.H4(
                "Estações monitoradas no estado"
            ),

            dash_table.DataTable(

                columns=[
                    {
                        "name":
                            c,
                        "id":
                            c,
                    }
                    for c
                    in tabela.columns
                ],

                data=
                    tabela.to_dict(
                        "records"
                    ),

                page_size=
                    15,

                sort_action=
                    "native",

                filter_action=
                    "native",

                style_table={
                    "overflowX":
                        "auto",
                },

                style_cell={
                    "textAlign":
                        "left",
                    "padding":
                        "6px",
                    "fontSize":
                        "11px",
                },

                style_header={
                    "fontWeight":
                        "bold",
                    "backgroundColor":
                        "#f0f0f0",
                },
            ),

            (
                html.Div(
                    f"Estações sem classificação na rede principal: {n_sem_class:,}.",
                    style={
                        "fontSize":
                            "11px",
                        "color":
                            "#777",
                        "marginTop":
                            "8px",
                    },
                )
                if n_sem_class > 0
                else
                html.Div()
            ),
        ]
    )


# =============================================================================
# COTAGRAMA
# =============================================================================

@app.callback(

    Output(
        "cotagrama",
        "figure",
    ),

    Input(
        "estacao",
        "value",
    ),

    Input(
        "periodo",
        "value",
    ),
)

def atualizar_cotagrama(
    codigo,
    periodo,
):

    # ANA: apresenta a climatologia mensal calculada com as médias
    # mensais anuais e destaca a cota atual no mês correspondente.
    if str(
        codigo
    ).startswith(
        "ANA:"
    ):

        meta_ana = estacoes_geo[
            estacoes_geo[
                "codigo"
            ]
            ==
            str(
                codigo
            )
        ]

        if meta_ana.empty:

            return go.Figure()

        linha_ana = meta_ana.iloc[
            0
        ]

        codigo_original = texto_valido(
            linha_ana.get(
                "codigo_original",
                "",
            )
        )

        fig = go.Figure()

        if not ARQ_ANA_REFERENCIA_MENSAL.exists():

            fig.add_annotation(
                text=
                    "Arquivo de referência histórica mensal da ANA não encontrado.",
                x=
                    0.5,
                y=
                    0.5,
                showarrow=
                    False,
            )

            return fig

        ref_est = pd.read_parquet(
            ARQ_ANA_REFERENCIA_MENSAL
        ).copy()

        ref_est[
            "codigo"
        ] = (
            ref_est[
                "codigo"
            ]
            .astype(str)
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
            .str.strip()
        )

        ref_est = ref_est[
            ref_est[
                "codigo"
            ]
            ==
            codigo_original
        ].copy()

        if ref_est.empty:

            fig.add_annotation(
                text=
                    "Sem referência histórica mensal disponível para esta estação.",
                x=
                    0.5,
                y=
                    0.5,
                showarrow=
                    False,
            )

            return fig

        ref_est = ref_est.sort_values(
            "mes"
        )

        meses_ordem = [
            "Jan",
            "Fev",
            "Mar",
            "Abr",
            "Mai",
            "Jun",
            "Jul",
            "Ago",
            "Set",
            "Out",
            "Nov",
            "Dez",
        ]

        mapa_mes = {
            1: "Jan",
            2: "Fev",
            3: "Mar",
            4: "Abr",
            5: "Mai",
            6: "Jun",
            7: "Jul",
            8: "Ago",
            9: "Set",
            10: "Out",
            11: "Nov",
            12: "Dez",
        }

        ref_est[
            "mes_rotulo"
        ] = ref_est[
            "mes"
        ].map(
            mapa_mes
        )

        # Máxima mensal histórica
        fig.add_trace(
            go.Scatter(
                x=
                    ref_est[
                        "mes_rotulo"
                    ],
                y=
                    ref_est[
                        "maior_media_mensal_m"
                    ],
                name=
                    "Maior média mensal",
                legendrank=
                    4,
                mode=
                    "lines+markers",
                line={
                    "dash":
                        "dot",
                    "width":
                        1.5,
                    "color":
                        "#5c7cfa",
                },
                marker={
                    "size":
                        6,
                },
                hovertemplate=(
                    "%{x}<br>"
                    "Maior média mensal: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        # Mínima mensal histórica + faixa
        fig.add_trace(
            go.Scatter(
                x=
                    ref_est[
                        "mes_rotulo"
                    ],
                y=
                    ref_est[
                        "menor_media_mensal_m"
                    ],
                name=
                    "Menor média mensal",
                legendrank=
                    3,
                mode=
                    "lines+markers",
                fill=
                    "tonexty",
                fillcolor=
                    "rgba(130,130,130,0.10)",
                line={
                    "dash":
                        "dot",
                    "width":
                        1.5,
                    "color":
                        "#ff6b6b",
                },
                marker={
                    "size":
                        6,
                },
                hovertemplate=(
                    "%{x}<br>"
                    "Menor média mensal: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        # Média mensal histórica
        fig.add_trace(
            go.Scatter(
                x=
                    ref_est[
                        "mes_rotulo"
                    ],
                y=
                    ref_est[
                        "media_mensal_historica_m"
                    ],
                name=
                    "Média histórica mensal",
                legendrank=
                    2,
                mode=
                    "lines+markers",
                line={
                    "dash":
                        "dash",
                    "width":
                        2,
                    "color":
                        "#12b886",
                },
                marker={
                    "size":
                        7,
                },
                hovertemplate=(
                    "%{x}<br>"
                    "Média histórica mensal: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        cota_atual = pd.to_numeric(
            linha_ana.get(
                "cotaUltimaMedicao_m",
                np.nan,
            ),
            errors=
                "coerce",
        )

        data_atual = pd.to_datetime(
            linha_ana.get(
                "dataHoraUltimaMedicao",
                pd.NaT,
            ),
            errors=
                "coerce",
        )

        mes_atual = (
            int(
                data_atual.month
            )
            if pd.notna(
                data_atual
            )
            else
            int(
                linha_ana.get(
                    "mes_referencia_atual",
                    1,
                )
            )
        )

        mes_atual_rotulo = mapa_mes.get(
            mes_atual,
            "",
        )

        if (
            pd.notna(
                cota_atual
            )
            and
            mes_atual_rotulo
        ):

            fig.add_trace(
                go.Scatter(
                    x=[
                        mes_atual_rotulo
                    ],
                    y=[
                        cota_atual
                    ],
                    mode=
                        "markers",
                    name=
                        "Cota atual",
                    legendrank=
                        1,
                    marker={
                        "size":
                            15,
                        "symbol":
                            "diamond",
                        "color":
                            "#9c5cff",
                    },
                    hovertemplate=(
                        "Cota atual: %{y:.2f} m"
                        "<extra></extra>"
                    ),
                )
            )

        nome_mes = texto_valido(
            linha_ana.get(
                "nome_mes",
                "",
            )
        )

        periodo_ref = texto_valido(
            linha_ana.get(
                "periodo_referencia",
                "",
            )
        )

        n_anos = pd.to_numeric(
            linha_ana.get(
                "n_anos_validos",
                np.nan,
            ),
            errors=
                "coerce",
        )

        fig.update_layout(
            title=
                f"Cotagrama de referência mensal — {linha_ana['nome']}",
            template=
                "plotly_white",
            height=
                570,
            hovermode=
                "x unified",
            xaxis_title=
                "Mês",
            yaxis_title=
                "Cota (m)",
            xaxis={
                "categoryorder":
                    "array",
                "categoryarray":
                    meses_ordem,
            },
            legend={
                "orientation":
                    "h",
                "y":
                    1.13,
            },
            margin={
                "t":
                    85,
                "l":
                    60,
                "r":
                    40,
                "b":
                    50,
            },
        )

        return fig


    meta = estacoes_geo[
        estacoes_geo[
            "codigo"
        ]
        ==
        str(
            codigo
        )
    ]


    if meta.empty:

        return go.Figure()


    linha = meta.iloc[
        0
    ]


    serie = historico[
        historico[
            "codigo"
        ]
        ==
        str(
            codigo
        )
    ].copy()


    serie = serie.sort_values(
        "data"
    )


    if serie.empty:

        fig = go.Figure()

        fig.add_annotation(

            text=
                "Sem série histórica disponível.",

            x=
                0.5,

            y=
                0.5,

            showarrow=
                False,
        )

        return fig


    fim = serie[
        "data"
    ].max()


    if periodo == "30d":

        inicio = (
            fim
            -
            pd.DateOffset(
                days=30
            )
        )


    elif periodo == "3m":

        inicio = (
            fim
            -
            pd.DateOffset(
                months=3
            )
        )


    elif periodo == "6m":

        inicio = (
            fim
            -
            pd.DateOffset(
                months=6
            )
        )


    elif periodo == "1a":

        inicio = (
            fim
            -
            pd.DateOffset(
                years=1
            )
        )


    else:

        inicio = serie[
            "data"
        ].min()


    plot = serie[
        serie[
            "data"
        ]
        >=
        inicio
    ].copy()


    fig = go.Figure()


    # =========================================================================
    # MÁXIMA
    # =========================================================================

    fig.add_trace(

        go.Scatter(

            x=
                plot[
                    "data"
                ],

            y=
                plot[
                    "maximo_m"
                ],

            name=
                "Máxima histórica",

            mode=
                "lines",

            line={

                "dash":
                    "dot",

                "width":
                    1.5,

                "color":
                    "#5c7cfa",
            },
        )
    )


    # =========================================================================
    # MÍNIMA
    # =========================================================================

    fig.add_trace(

        go.Scatter(

            x=
                plot[
                    "data"
                ],

            y=
                plot[
                    "minimo_m"
                ],

            name=
                "Mínima histórica",

            mode=
                "lines",

            fill=
                "tonexty",

            fillcolor=
                "rgba(130,130,130,0.10)",

            line={

                "dash":
                    "dot",

                "width":
                    1.5,

                "color":
                    "#ff6b6b",
            },
        )
    )


    # =========================================================================
    # MÉDIA
    # =========================================================================

    fig.add_trace(

        go.Scatter(

            x=
                plot[
                    "data"
                ],

            y=
                plot[
                    "media_m"
                ],

            name=
                "Média histórica",

            mode=
                "lines",

            line={

                "dash":
                    "dash",

                "width":
                    2,

                "color":
                    "#12b886",
            },
        )
    )


    # =========================================================================
    # OBSERVADA
    # =========================================================================

    fig.add_trace(

        go.Scatter(

            x=
                plot[
                    "data"
                ],

            y=
                plot[
                    "atual_m"
                ],

            name=
                "Cota observada",

            mode=
                "lines",

            line={

                "width":
                    3,

                "color":
                    "#9c5cff",
            },
        )
    )


    # =========================================================================
    # LIMIARES
    # =========================================================================

    limites = [

        (
            "cotaAtencaoEstiagem_m",
            "Atenção — estiagem",
        ),

        (
            "cotaAtencaoInundacao_m",
            "Atenção — inundação",
        ),

        (
            "cotaAlerta_m",
            "Alerta",
        ),

        (
            "cotaInundacao_m",
            "Inundação",
        ),
    ]


    for campo, rotulo in limites:

        valor = linha.get(
            campo,
            np.nan,
        )


        if pd.isna(
            valor
        ):

            continue


        fig.add_hline(

            y=
                float(
                    valor
                ),

            line_dash=
                "dash",

            line_width=
                1.2,

            annotation_text=
                rotulo,

            annotation_position=
                "top left",
        )


    fig.update_layout(

        title=
            f"Cotagrama — {linha['nome']}",

        template=
            "plotly_white",

        height=
            570,

        hovermode=
            "x unified",

        xaxis_title=
            "Data",

        yaxis_title=
            "Cota (m)",

        legend={

            "orientation":
                "h",

            "y":
                1.13,
        },

        margin={
            "t": 85,
            "l": 60,
            "r": 40,
            "b": 50,
        },
    )


    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print(
        "\n"
        +
        "#" * 80
    )

    print(
        "ETAPA 04 PRONTA"
    )

    print(
        "CONDIÇÃO HIDROLÓGICA + REDE HIDROGRÁFICA + SAÚDE"
    )

    print(
        "#" * 80
    )


    print(
        f"\nEstações: "
        f"{len(estacoes_geo):,}"
    )

    print(
        f"UBS: "
        f"{len(ubs):,}"
    )

    print(
        f"UBSI: "
        f"{len(ubsi):,}"
    )

    print(
        f"UPA: "
        f"{len(upa):,}"
    )

    print(
        f"Polos Base: "
        f"{len(POLOS):,}"
    )


    print(
        "\nNeste computador:"
    )

    print(
        f"http://127.0.0.1:{PORT}"
    )


    print(
        "\nNa rede:"
    )

    print(
        f"http://10.1.244.182:{PORT}"
    )


    print(
        "\n"
    )


    app.run(

        debug=
            False,

        host=
            HOST,

        port=
            PORT,

        threaded=
            True,
    )
