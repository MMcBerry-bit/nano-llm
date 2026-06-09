import { Router, type IRouter } from "express";
import healthRouter from "./health";
import modelsRouter from "./models";

const router: IRouter = Router();

router.use(healthRouter);
router.use(modelsRouter);

export default router;
