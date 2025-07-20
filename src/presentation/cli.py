"""
Command-line interface for Zero-Click Compass using SOLID architecture.
Following Single Responsibility and Dependency Inversion principles.
"""
import argparse
import sys
from typing import Dict, Any

from ..core.container import configure_services, get_service
from ..core.interfaces import ConfigurationProvider, Logger
from ..application.pipeline import PipelineService
from ..infrastructure.config import get_default_configuration


class CLICommand:
    """Base class for CLI commands."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def execute(self, args: argparse.Namespace) -> None:
        """Execute the command."""
        raise NotImplementedError


class PipelineCommand(CLICommand):
    """Command for running the complete pipeline."""
    
    def __init__(self, pipeline_service: PipelineService, logger: Logger):
        super().__init__(logger)
        self.pipeline_service = pipeline_service
    
    def execute(self, args: argparse.Namespace) -> None:
        """Execute the pipeline command."""
        self.logger.info(f"Starting pipeline for {args.url} with query: {args.query}")
        
        config_overrides = {
            'max_pages': args.max_pages,
            'max_expansions': args.max_expansions,
            'top_k': args.top_k,
            'include_social': args.social
        }
        
        try:
            analysis = self.pipeline_service.run_analysis(
                args.url, 
                args.query, 
                config_overrides
            )
            
            # Display results
            self._display_results(analysis)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            sys.exit(1)
    
    def _display_results(self, analysis) -> None:
        """Display analysis results."""
        self.logger.info("=== ANALYSIS RESULTS ===")
        self.logger.info(f"Total chunks: {len(analysis.chunks)}")
        self.logger.info(f"Query coverage: {analysis.calculate_coverage():.2%}")
        
        best_chunks = analysis.get_best_performing_chunks(5)
        self.logger.info(f"Top {len(best_chunks)} performing chunks:")
        
        for i, chunk in enumerate(best_chunks, 1):
            self.logger.info(f"{i}. {chunk.get_text_preview()} (URL: {chunk.url})")


class ConfigCommand(CLICommand):
    """Command for configuration management."""
    
    def __init__(self, config: ConfigurationProvider, logger: Logger):
        super().__init__(logger)
        self.config = config
    
    def execute(self, args: argparse.Namespace) -> None:
        """Execute the config command."""
        if args.action == 'show':
            self._show_config()
        elif args.action == 'set':
            self._set_config(args.key, args.value)
        elif args.action == 'get':
            self._get_config(args.key)
    
    def _show_config(self) -> None:
        """Show current configuration."""
        self.logger.info("Current configuration:")
        # This would show current config values
        self.logger.info("Use 'config get <key>' to get specific values")
    
    def _set_config(self, key: str, value: str) -> None:
        """Set configuration value."""
        self.config.set(key, value)
        self.logger.info(f"Set {key} = {value}")
    
    def _get_config(self, key: str) -> None:
        """Get configuration value."""
        value = self.config.get(key)
        self.logger.info(f"{key} = {value}")


class StatusCommand(CLICommand):
    """Command for checking system status."""
    
    def execute(self, args: argparse.Namespace) -> None:
        """Execute the status command."""
        self.logger.info("=== SYSTEM STATUS ===")
        
        # Check API keys
        config = get_service(ConfigurationProvider)
        
        google_key = config.get('GOOGLE_API_KEY')
        if google_key:
            self.logger.info("✅ Google API key configured")
        else:
            self.logger.info("❌ Google API key missing")
        
        reddit_id = config.get('REDDIT_CLIENT_ID')
        if reddit_id:
            self.logger.info("✅ Reddit API configured")
        else:
            self.logger.info("❌ Reddit API not configured")
        
        twitter_token = config.get('TWITTER_BEARER_TOKEN')
        if twitter_token:
            self.logger.info("✅ Twitter API configured")
        else:
            self.logger.info("❌ Twitter API not configured")


class ZeroClickCompassCLI:
    """Main CLI application using SOLID principles."""
    
    def __init__(self):
        self.commands: Dict[str, CLICommand] = {}
        self._configure_system()
        self._register_commands()
    
    def _configure_system(self) -> None:
        """Configure the dependency injection container."""
        config = get_default_configuration()
        configure_services(config)
    
    def _register_commands(self) -> None:
        """Register all available commands."""
        # Get services from DI container
        logger = get_service(Logger)
        config = get_service(ConfigurationProvider)
        pipeline_service = get_service(PipelineService)
        
        # Register commands
        self.commands['pipeline'] = PipelineCommand(pipeline_service, logger)
        self.commands['config'] = ConfigCommand(config, logger)
        self.commands['status'] = StatusCommand(logger)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="Zero-Click Compass - LLM-first website performance analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_examples()
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Pipeline command
        pipeline_parser = subparsers.add_parser('pipeline', help='Run complete analysis pipeline')
        pipeline_parser.add_argument('url', help='Website URL to analyze')
        pipeline_parser.add_argument('query', help='Analysis query')
        pipeline_parser.add_argument('--max-pages', type=int, default=50, 
                                   help='Maximum pages to crawl')
        pipeline_parser.add_argument('--max-expansions', type=int, default=15,
                                   help='Maximum query expansions')
        pipeline_parser.add_argument('--top-k', type=int, default=10,
                                   help='Number of top results')
        pipeline_parser.add_argument('--social', action='store_true',
                                   help='Include social media analysis')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Manage configuration')
        config_parser.add_argument('action', choices=['show', 'get', 'set'],
                                 help='Configuration action')
        config_parser.add_argument('--key', help='Configuration key')
        config_parser.add_argument('--value', help='Configuration value')
        
        # Status command
        subparsers.add_parser('status', help='Check system status')
        
        return parser
    
    def _get_examples(self) -> str:
        """Get CLI usage examples."""
        return """
Examples:
  # Run complete pipeline
  python -m src.presentation.cli pipeline https://example.com "marketing strategies"
  
  # Run with social media analysis
  python -m src.presentation.cli pipeline https://example.com "SEO tips" --social
  
  # Check system status
  python -m src.presentation.cli status
  
  # Configure API key
  python -m src.presentation.cli config set --key GOOGLE_API_KEY --value your_key
        """
    
    def run(self, args: list = None) -> None:
        """Run the CLI application."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            sys.exit(1)
        
        try:
            command = self.commands[parsed_args.command]
            command.execute(parsed_args)
        except KeyError:
            print(f"Unknown command: {parsed_args.command}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli = ZeroClickCompassCLI()
    cli.run()


if __name__ == "__main__":
    main() 