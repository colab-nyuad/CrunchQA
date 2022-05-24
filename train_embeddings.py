from pykeen.triples import TriplesFactory, TriplesNumericLiteralsFactory
from pykeen.pipeline import pipeline
import argparse

parser = argparse.ArgumentParser(
    description="Graph Embedding for Question Answering over Knowledge Graphs"
)

parser.add_argument(
    "--train_path", type=str, help="Path to training data"
)

parser.add_argument(
    "--valid_path", type=str, help="Path to valid data"
)

parser.add_argument(
    "--test_path", type=str, help="Path to test data"
)

parser.add_argument(
    "--train_path_literals", type=str, default="", help="Path to training data literals"
)

parser.add_argument(
    "--valid_path_literals", type=str, default="", help="Path to valid data literals"
)

parser.add_argument(
    "--test_path_literals", type=str, default="", help="Path to test data literals"
)

parser.add_argument(
    "--model", default="ComplEx", help="Embedding model"
)

parser.add_argument(
    "--loss", type=str, help="Loss function"
)

parser.add_argument(
    '--create_inverse_triples', type=bool, default=True, help = "Create inverse triples"
)

parser.add_argument(
    '--gpu', type=int, default=0, help = "Which gpu to use"
)

parser.add_argument(
    '--batch_size', type=int, default=2048, help = "Batch size"
)

parser.add_argument(
    '--num_epochs', type=int, default=30, help = "NUmber of epochs"
)

parser.add_argument(
    '--dim', type=int, default=200, help = "Dimension"
)

parser.add_argument(
    "--results_folder", type=str, help="Results folder"
)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.train_path_literals == "":
        training = TriplesFactory.from_path(args.train_path, 
                                          create_inverse_triples=args.create_inverse_triples)
        validation = TriplesFactory.from_path(args.valid_path,
                                          entity_to_id=training.entity_to_id,
                                          relation_to_id=training.relation_to_id,
                                          create_inverse_triples=args.create_inverse_triples)

        testing = TriplesFactory.from_path(args.test_path,
                                       entity_to_id=training.entity_to_id,
                                       relation_to_id=training.relation_to_id,
                                       create_inverse_triples=args.create_inverse_triples)

    else:
        training = TriplesNumericLiteralsFactory(path=args.train_path, path_to_numeric_triples=args.train_path_literals)
        validation = TriplesNumericLiteralsFactory(path=args.valid_path,
                                                   path_to_numeric_triples=args.valid_path_literals,
                                                   entity_to_id=training.entity_to_id,
                                                   relation_to_id=training.relation_to_id)

        testing = TriplesNumericLiteralsFactory(path=args.test_path,
                                                path_to_numeric_triples=args.test_path_literals,
                                                entity_to_id=training.entity_to_id,
                                                relation_to_id=training.relation_to_id)

    result = pipeline(training=training, testing=testing, validation=validation,
                      model=args.model, model_kwargs=dict(embedding_dim=args.dim),
                      device='cuda:{}'.format(args.gpu),loss=args.loss,
                      training_kwargs=dict(num_epochs=args.num_epochs,
                                           batch_size=args.batch_size,
                                           checkpoint_name='checkpoint_best.pt',
                                           checkpoint_frequency=5,
                                           checkpoint_directory=args.results_folder),
                      evaluator_kwargs=dict(filtered=True))

    result.save_to_directory(args.results_folder)
