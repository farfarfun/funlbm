import typer

funlbm_cli = typer.Typer(help="funlbm")


def funlbm() -> None:
    """funlbm 命令行入口，分发到 run / submit / update 等子命令"""
    funlbm_cli()
